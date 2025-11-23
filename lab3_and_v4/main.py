import argparse
import keras
import numpy as np
from data import load_data
from train import train_models, tune_model_v2

parser = argparse.ArgumentParser()
parser.add_argument("--train", action="store_true")
parser.add_argument("--predict", choices=["classic", "tuned"])
parser.add_argument("--model", type=str, default="v2", choices=["v1", "v2"])
parser.add_argument(
    "--features",
    type=float,
    nargs=13,
    default=[14.13, 4.1, 2.74, 24.5, 96, 2.05, 0.76, 0.56, 1.35, 9.2, 0.61, 1.6, 560],
)
parser.add_argument("--epochs", type=int, default=30)
parser.add_argument("--batch_size", type=int, default=8)
parser.add_argument("--tune", action="store_true")
args = parser.parse_args()

if args.train:
    x_train, x_test, y_train, y_test, mean, std = load_data()
    print(f"Training data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    train_models(x_train, x_test, y_train, y_test, args.epochs, args.batch_size)

elif args.tune:
    x_train, x_test, y_train, y_test, mean, std = load_data()
    print(f"Tuning data: {x_train.shape}")
    print(f"Test data: {x_test.shape}")
    tune_model_v2(x_train, x_test, y_train, y_test)

    tuned_model = keras.models.load_model("tuned_model_improved_v2.keras")
    _, tuned_acc = tuned_model.evaluate(x_test, y_test, verbose=0)
    print(f"Tuned model accuracy: {tuned_acc:.4f}")

    try:
        model_v1 = keras.models.load_model("model_improved_v1.keras")
        _, acc_v1 = model_v1.evaluate(x_test, y_test, verbose=0)
        print(f"Baseline V1 accuracy: {acc_v1:.4f}")
    except Exception:
        print("Baseline V1 not found")
        acc_v1 = 0
    try:
        model_v2 = keras.models.load_model("model_improved_v2.keras")
        _, acc_v2 = model_v2.evaluate(x_test, y_test, verbose=0)
        print(f"Baseline V2 accuracy: {acc_v2:.4f}")
    except Exception:
        print("Baseline V2 not found")
        acc_v2 = 0

    if tuned_acc > max(acc_v1, acc_v2):
        print(f"Tuned model is the best (accuracy: {tuned_acc:.4f})")
    elif acc_v2 > acc_v1:
        print(f"Baseline V2 is the best (accuracy: {acc_v2:.4f})")
    elif acc_v1 > acc_v2:
        print(f"Baseline V1 is the best (accuracy: {acc_v1:.4f})")
    else:
        print("Models have similar accuracy")

elif args.predict:
    _, _, _, _, mean, std = load_data()

    # (13,) to (1, 13) because model expects batch dimension
    if len(args.features) != 13:
        print("Error: provide 13 features for prediction")
        exit(1)
    features = np.array(args.features, dtype=np.float32).reshape(1, -1)
    features = (features - mean) / (std + 1e-8)

    try:
        if args.predict == "classic":
            model = keras.models.load_model(f"model_improved_{args.model}.keras")
        else:
            model = keras.models.load_model("tuned_model_improved_v2.keras")
        model.summary()
        prediction = model.predict(features)
        wine_class = np.argmax(prediction) + 1

        print(f"Wine class: {wine_class}")
    except Exception as e:
        print(f"Error: train model first ({e})")
