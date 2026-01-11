import os
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf


def make_predictions(model, last_sequence, n_predictions):
    predictions = []
    current_sequence = last_sequence.copy()

    for step in range(n_predictions):
        next_prediction = model.predict(
            current_sequence[np.newaxis, :],
            verbose=0,
        )[0, 0]

        predictions.append(next_prediction)
        new_row = current_sequence[-1].copy()
        new_row[0] = next_prediction

        time_index = len(current_sequence) + step
        new_row[1] = np.sin(2 * np.pi * time_index / 24)
        new_row[2] = np.cos(2 * np.pi * time_index / 24)

        current_sequence = np.vstack([current_sequence[1:], new_row])

    return np.array(predictions)


def predict(
    history_file,
    model_type="lstm",
    model_dir="./models",
    n_steps=10,
    output_file="predictions.csv",
):
    """Make predictions using trained model."""

    processor_path = os.path.join(model_dir, "processor.pkl")
    try:
        with open(processor_path, "rb") as f:
            processor = pickle.load(f)
    except FileNotFoundError:
        print(
            f"Error: {processor_path} not found. Train first with: python train.py --data <file>"
        )
        return

    model_filename = f"{model_type}_model.keras"
    model_path = os.path.join(model_dir, model_filename)

    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        print(
            f"Error: {model_path} not found. Train first with: python train.py --data <file>"
        )
        return

    try:
        data = processor.load_csv(history_file)
    except FileNotFoundError:
        print(f"Error: {history_file} not found")
        return

    data_enhanced = processor.add_harmonics(data, period=24)
    last_sequence = processor.get_last_sequence(data_enhanced)

    print(f"Forecasting {n_steps} steps with {model_type.upper()}...")
    predictions_normalized = make_predictions(model, last_sequence, n_steps)
    predictions_original_scale = processor.inverse_transform(predictions_normalized)

    results_df = pd.DataFrame(
        {"step": np.arange(1, n_steps + 1), "prediction": predictions_original_scale}
    )

    results_df.to_csv(output_file, index=False)
    print(f"Results saved to: {output_file}")
    print(results_df)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Make predictions with trained model")
    parser.add_argument("--history", type=str, required=True, help="CSV history file")
    parser.add_argument("--n", type=int, default=10, help="Steps to forecast")
    parser.add_argument(
        "--output", type=str, default="predictions.csv", help="Output file"
    )
    parser.add_argument(
        "--model-dir", type=str, default="./models", help="Model directory"
    )
    parser.add_argument(
        "--model-type", type=str, default="lstm", choices=["lstm", "dense"]
    )

    args = parser.parse_args()

    predict(
        history_file=args.history,
        model_type=args.model_type,
        model_dir=args.model_dir,
        n_steps=args.n,
        output_file=args.output,
    )
