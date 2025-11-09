import argparse
import keras
import numpy as np
from data import load_data
from train import train_models

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", action="store_true")
parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2"])
parser.add_argument(
    "--features",
    type=float,
    nargs=13,
    default=[14.13, 4.1, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560],
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
args = parser.parse_args()

if args.train:
    x_train, x_test, y_train, y_test, mean, std = load_data()
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    train_models(x_train, x_test, y_train, y_test, args.epochs, args.batch_size)

elif args.predict:
    _, _, _, _, mean, std = load_data()

    # (13,) to (1, 13) because model expects batch dimension
    if len(args.features) != 13:
        print("Error: provide 13 features for prediction")
        exit(1)
    features = np.array(args.features, dtype=np.float32).reshape(1, -1)
    features = (features - mean) / (std + 1e-8)

    try:
        model = keras.models.load_model(f"model_{args.model}.keras")
        prediction = model.predict(features)
        wine_class = np.argmax(prediction) + 1
        print(f"Wine class: {wine_class}")
    except Exception as e:
        print(f"Error: train model first ({e})")
