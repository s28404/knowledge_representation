import os
import pickle
import argparse
import numpy as np
import pandas as pd
import tensorflow as tf

from train import train_models
from predict import predict


def make_predictions(model, last_sequence, n_predictions):
    predictions = []
    # use a copy to avoid modifying the original last_sequence
    current_sequence = last_sequence.copy()

    for step in range(n_predictions):
        next_prediction = model.predict(
            # we do [np.newaxis, :] to add batch dimension from [time_steps, features] to [1, time_steps, features]
            # because model.predict expects batch dimension
            current_sequence[np.newaxis, :],
            verbose=0,
        )[0, 0]
        # next_prediction.shape before [0, 0] is (1, 1), after indexing it's a scalar

        predictions.append(next_prediction)
        # current_sequence shape is (time_steps, features)
        # [-1] gets the last row (step) with shape (features,)
        new_row = current_sequence[-1].copy()
        # take [0] instead of [1] because [0] is the target variable, [1] and [2] are harmonics
        new_row[0] = next_prediction

        time_index = len(current_sequence) + step
        # new[1] and new[2] are harmonics (sin and cos components for daily seasonality)
        # we update them based on the new time index
        new_row[1] = np.sin(2 * np.pi * time_index / 24)
        new_row[2] = np.cos(2 * np.pi * time_index / 24)

        # current_sequence shape is (time_steps, features), current_sequence[1:] removes the oldest step
        # because we append new_row at the end, the shape remains the same
        # we use vstack to stack arrays in sequence vertically (row wise)
        current_sequence = np.vstack([current_sequence[1:], new_row])

    return np.array(predictions)


def main():
    parser = argparse.ArgumentParser(description="Time series forecasting")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # Train command
    train_parser = subparsers.add_parser("train", help="Train models")
    train_parser.add_argument(
        "--data", type=str, required=True, help="CSV file with training data"
    )
    train_parser.add_argument(
        "--output", type=str, default="./models", help="Output directory"
    )

    # Predict command
    predict_parser = subparsers.add_parser("predict", help="Make predictions")
    predict_parser.add_argument(
        "--history", type=str, required=True, help="CSV history file"
    )
    predict_parser.add_argument("--n", type=int, default=10, help="Steps to forecast")
    predict_parser.add_argument(
        "--result", type=str, default="predictions.csv", help="Output file"
    )
    predict_parser.add_argument(
        "--model-dir", type=str, default="./models", help="Model directory"
    )
    predict_parser.add_argument(
        "--model-type", type=str, default="lstm", choices=["lstm", "dense"]
    )

    args = parser.parse_args()

    if args.command == "train":
        train_models(args.data, args.output)
    elif args.command == "predict":
        predict(args)
    else:
        parser.print_help()


def predict(args):
    processor_path = os.path.join(args.model_dir, "processor.pkl")
    try:
        with open(processor_path, "rb") as f:
            processor = pickle.load(f)
    except FileNotFoundError:
        print(f"Error: {processor_path} not found")
        return

    model_filename = f"{args.model_type}_model.keras"
    model_path = os.path.join(args.model_dir, model_filename)

    try:
        model = tf.keras.models.load_model(model_path)
    except FileNotFoundError:
        print(f"Error: {model_path} not found")
        return

    try:
        data = processor.load_csv(args.history)
    except FileNotFoundError:
        print(f"Error: {args.history} not found")
        return

    # perdiod=24 gives daily seasonality harmonics
    # data_enhanced shape is (time_steps, features + 2) because we add 2 harmonics (sin and cos components for daily seasonality)
    data_enhanced = processor.add_harmonics(data, period=24)
    # we put last sequence for forecasting, get_last_sequence returns the last lookback_window steps
    last_sequence = processor.get_last_sequence(data_enhanced)

    print(f"Forecasting {args.n} steps with {args.model_type.upper()}...")
    # args.n is number of steps to forecast for example 10 steps makes 10 predictions ahead
    # in make_predictions we normalize the data before feeding into model because model was trained on normalized data
    predictions_normalized = make_predictions(model, last_sequence, args.n)
    # inverse transform to get back to original scale from normalized scale
    predictions_original_scale = processor.inverse_transform(predictions_normalized)

    results_df = pd.DataFrame(
        {"step": np.arange(1, args.n + 1), "prediction": predictions_original_scale}
    )

    # index=False to avoid writing row numbers to CSV
    results_df.to_csv(args.result, index=False)
    print(f"Results saved to: {args.result}")
    print(results_df)


if __name__ == "__main__":
    main()
