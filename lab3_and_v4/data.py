import pandas as pd
import numpy as np


def load_data():
    # shape: (178, 14)
    data = pd.read_csv("data.csv")

    # y.shape: (178,) (labels)
    y = data.iloc[:, 0].values
    # x.shape: (178, 13) (features)
    x = data.iloc[:, 1:].values

    idx = np.random.permutation(len(x))
    x = x[idx]
    y = y[idx]

    # shape: [num_samples, num_features]
    # features will be normalized by the model layer
    mean = x.mean(axis=0)
    std = x.std(axis=0)
    # x remains unnormalized

    y_hot = np.zeros((len(y), 3))
    for i in range(len(y)):
        y_hot[i, int(y[i]) - 1] = 1

    split = int(len(x) * 0.8)
    return (
        x[:split].astype(np.float32),
        x[split:].astype(np.float32),
        y_hot[:split].astype(np.float32),
        y_hot[split:].astype(np.float32),
        mean,
        std,
    )
