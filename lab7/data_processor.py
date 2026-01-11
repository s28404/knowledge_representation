import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


class DataProcessor:
    # Processor is a class to handle data loading, preprocessing, and sequence creation
    # lookback_window: number of past time steps to consider for each input sequence
    # scale: whether to apply Min-Max scaling to the data
    # MinMaxScaler scales each feature to a given range, default is (0, 1) from (min, max)
    # we want use it to normalize the data before feeding into neural network models
    # it is useful because neural networks perform better with normalized data
    def __init__(self, lookback_window=24, scale=True):
        self.lookback_window = lookback_window
        self.scale = scale
        if scale:
            self.scaler = MinMaxScaler(feature_range=(0, 1))
        else:
            self.scaler = None
        self.data_shape = None
        self.original_data = None

    def load_csv(self, filepath, target_column=None):
        df = pd.read_csv(filepath)
        # Select only numeric columns or the specified target column
        # [np.number] selects all numeric data types
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        # target_column is the name of the column we want to forecast
        df = df[[target_column]] if target_column else df[numeric_cols]
        data = df.values
        # we use copy to avoid modifying the original data outside this class
        self.original_data = data.copy()
        if self.scale:
            # scaler.fit_transform computes min and max values and scales the data
            # fit_transform fits the scaler to the data and transforms it in one step
            data = self.scaler.fit_transform(data)
        self.data_shape = data.shape
        return data

    # period=24 adds daily seasonality harmonics (sin and cos components for daily seasonality)
    def add_harmonics(self, data, period=24):
        # data.shape is (n_samples, n_features)
        # where n_samples is the number of time steps, n_features is the number of features
        n_samples = data.shape[0]
        # from (n_samples, n_features) to (n_samples, n_features + 2)
        indices = np.arange(n_samples).reshape(-1, 1)
        # sin_harm.shape (n_samples, 1)
        sin_harm = np.sin(2 * np.pi * indices / period)
        # cos_harm.shape (n_samples, 1)
        cos_harm = np.cos(2 * np.pi * indices / period)
        # enhanced_data.shape (n_samples, n_features + 2)
        enhanced_data = np.hstack([data, sin_harm, cos_harm])
        return enhanced_data

    # target_idx=0 specifies which column is the target variable
    def create_sequences(self, data, target_idx=0):
        X = []
        y = []
        for i in range(len(data) - self.lookback_window):
            # windows.shape (lookback_window, n_features) contains a sequence of past time steps like: data[i], data[i+1], ..., data[i+lookback_window-1]
            window = data[i : i + self.lookback_window]
            X.append(window)
            # target_value.shape () is a single value data[i + lookback_window][target_idx]
            target_value = data[i + self.lookback_window, target_idx]
            y.append(target_value)
        # X.shape (n_samples, lookback_window, n_features) is the input sequences
        # y.shape (n_samples,) is the target values
        return np.array(X), np.array(y)

    def inverse_transform(self, data):
        # we need it to restore original scale from normalized scale
        if self.scaler is None:
            return data
        # dummy.shape (data.shape[0], n_features) to match scaler's expected input shape
        dummy = np.zeros((data.shape[0], self.data_shape[1]))
        # from (n_samples,) to (n_samples, n_features)
        # [:, 0] makes sure we put data in the first column
        dummy[:, 0] = data
        # use scaler.inverse_transform to restore original scale
        restored = self.scaler.inverse_transform(dummy)
        # restored.shape (n_samples, n_features), we return only the first column
        return restored[:, 0]

    def get_last_sequence(self, data):
        # take the last lookback_window steps from data
        return data[-self.lookback_window :]
