import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers


def create_lstm_model(input_shape, lstm_units=32, dropout_rate=0.3):
    model = keras.Sequential()
    model.add(
        layers.LSTM(
            units=lstm_units,
            activation="relu",
            input_shape=input_shape,
            return_sequences=True,
        )
    )
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.LSTM(units=lstm_units // 2, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(16, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
        metrics=["mae"],
    )
    return model


def create_dense_model(input_shape, dense_units=64, dropout_rate=0.3):
    model = keras.Sequential()
    model.add(layers.Flatten(input_shape=input_shape))
    model.add(layers.Dense(units=dense_units, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(units=dense_units // 2, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(32, activation="relu"))
    model.add(layers.Dropout(dropout_rate))
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.005),
        loss="mse",
        metrics=["mae"],
    )
    return model


def create_tunable_lstm(hp, input_shape):
    model = keras.Sequential()
    model.add(
        layers.LSTM(
            units=hp.Int("lstm_units", min_value=16, max_value=128, step=16),
            activation="relu",
            input_shape=input_shape,
            return_sequences=True,
        )
    )
    model.add(
        layers.Dropout(hp.Float("dropout1", min_value=0.0, max_value=0.5, step=0.1))
    )
    model.add(
        layers.LSTM(
            units=hp.Int("lstm_units_2", min_value=8, max_value=64, step=8),
            activation="relu",
        )
    )
    model.add(
        layers.Dropout(hp.Float("dropout2", min_value=0.0, max_value=0.5, step=0.1))
    )
    model.add(
        layers.Dense(
            units=hp.Int("dense_units", min_value=8, max_value=32, step=8),
            activation="relu",
        )
    )
    model.add(layers.Dense(1))
    model.compile(
        optimizer=keras.optimizers.Adam(
            # sampling='log' means the learning rate will be sampled logarithmically between 1e-4 and 1e-2
            learning_rate=hp.Float("lr", min_value=1e-4, max_value=1e-2, sampling="log")
        ),
        loss="mse",
        metrics=["mae"],
    )
    return model
