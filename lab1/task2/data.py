import pandas as pd


def load_data(batch_size):
    splits = {'train': 'plain_text/train-00000-of-00001.parquet', 'test': 'plain_text/test-00000-of-00001.parquet'}
    train_set = pd.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["train"])
    test_set = pd.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["test"])

    df = pd.concat([train_set, test_set])
    
    data = df.to_numpy()
    print(data.shape)
    exit(1)

    data = data.reshape(-1, 32, 32, 3)

    return data

if __name__ == "__main__":
    data = load_data(32)
    print(data.shape)