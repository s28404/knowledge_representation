import os
import pickle
import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import tensorflow as tf
from sklearn.model_selection import train_test_split

from data_processor import DataProcessor
from models import create_lstm_model, create_dense_model


def plot_history(history, model_name, output_dir):
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(history.history["loss"], label="Train")
    plt.plot(history.history["val_loss"], label="Val")
    plt.title(f"{model_name} - Loss")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.subplot(1, 2, 2)
    plt.plot(history.history["mae"], label="Train")
    plt.plot(history.history["val_mae"], label="Val")
    plt.title(f"{model_name} - MAE")
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, f"{model_name}_history.png"), dpi=100)
    plt.close()


def train_models(data_file, output_dir="./models"):
    os.makedirs(output_dir, exist_ok=True)

    # Load and process data
    processor = DataProcessor(lookback_window=24, scale=True)
    data = processor.load_csv(data_file)
    data = processor.add_harmonics(data, period=24)

    X, y = processor.create_sequences(data, target_idx=0)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, shuffle=False
    )

    input_shape = (X_train.shape[1], X_train.shape[2])

    # Train LSTM
    print("Training LSTM...")
    lstm_model = create_lstm_model(input_shape)
    history_lstm = lstm_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                # patience=10 stops training if no improvement in val_loss for 10 epochs
                monitor="val_loss",
                patience=10,
                restore_best_weights=True,
            )
        ],
    )

    plot_history(history_lstm, "LSTM", output_dir)

    lstm_model.save(os.path.join(output_dir, "lstm_model.keras"))
    lstm_pred = lstm_model.predict(X_test, verbose=0).flatten()
    lstm_mse = np.mean((y_test - lstm_pred) ** 2)
    print(f"LSTM MSE: {lstm_mse:.6f}")

    # Train Dense
    print("Training Dense...")
    dense_model = create_dense_model(input_shape)
    history_dense = dense_model.fit(
        X_train,
        y_train,
        validation_split=0.2,
        epochs=50,
        batch_size=16,
        callbacks=[
            tf.keras.callbacks.EarlyStopping(
                monitor="val_loss", patience=10, restore_best_weights=True
            )
        ],
    )

    plot_history(history_dense, "Dense", output_dir)

    dense_model.save(os.path.join(output_dir, "dense_model.keras"))
    # dense.model.predict().shape is (n_samples, 1), we flatten it to (n_samples,)
    dense_pred = dense_model.predict(X_test, verbose=0).flatten()
    dense_mse = np.mean((y_test - dense_pred) ** 2)
    print(f"Dense MSE: {dense_mse:.6f}")

    # Save processor
    with open(os.path.join(output_dir, "processor.pkl"), "wb") as f:
        pickle.dump(processor, f)

    print(f"Models saved to {output_dir}")

    return {"lstm_mse": lstm_mse, "dense_mse": dense_mse}


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, required=True)
    parser.add_argument("--output", type=str, default="./models")
    args = parser.parse_args()
    train_models(args.data, args.output)
