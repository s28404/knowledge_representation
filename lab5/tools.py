import yaml
import matplotlib.pyplot as plt
import json


def load_hyperparameters(config_path="hyperparameters.yaml"):
    with open(config_path, "r") as file:
        config = yaml.safe_load(file)
    return config


def plot_both_types(save_path="plots/training_metrics_both.png"):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    model_types = ["dense", "cnn"]

    for i, model_type in enumerate(model_types):
        # Training metrics (left column)
        train_path = f"metrics/train_metrics_{model_type}.json"
        with open(train_path, "r") as file:
            train_metrics = json.load(file)

        # Test metrics (right column)
        test_path = f"metrics/test_metrics_{model_type}.json"
        with open(test_path, "r") as file:
            test_metrics = json.load(file)

        # Training Loss (top-left for dense, bottom-left for cnn)
        ax_train = axes[i, 0]
        epochs = range(1, len(train_metrics["loss"]) + 1)
        ax_train.plot(
            epochs, train_metrics["loss"], "b-", label="Training Loss", linewidth=2
        )
        ax_train.plot(
            epochs,
            train_metrics["val_loss"],
            "r-",
            label="Validation Loss",
            linewidth=2,
        )
        ax_train.set_title(f"{model_type.upper()} Model - Training Loss over Epochs")
        ax_train.set_xlabel("Epoch")
        ax_train.set_ylabel("Loss")
        ax_train.legend()
        ax_train.grid()

        # Training Accuracy (top-right for dense, bottom-right for cnn)
        ax_test = axes[i, 1]
        ax_test.plot(
            epochs,
            train_metrics["accuracy"],
            "b-",
            label="Training Accuracy",
            linewidth=2,
        )
        ax_test.plot(
            epochs,
            train_metrics["val_accuracy"],
            "r-",
            label="Validation Accuracy",
            linewidth=2,
        )
        ax_test.set_title(f"{model_type.upper()} Model - Training Accuracy over Epochs")
        ax_test.set_xlabel("Epoch")
        ax_test.set_ylabel("Accuracy")
        ax_test.set_ylim(0, 1)
        ax_test.legend()
        ax_test.grid()

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    print(f"Plot saved to {save_path}")


def plot(parameters="train_metrics.json", save_path="training_metrics.png"):
    with open(parameters, "r") as file:
        metrics = json.load(file)

    # Check if this is training history (loss is a list) or test metrics (loss is a float)
    is_history = isinstance(metrics.get("loss"), list)

    if is_history:
        # Training metrics with history curves
        fig, ax = plt.subplots(2, 2, figsize=(12, 10))

        epochs = range(1, len(metrics["loss"]) + 1)

        ax[0, 0].plot(epochs, metrics["loss"], "b-", label="Training Loss")
        ax[0, 0].plot(epochs, metrics["val_loss"], "r-", label="Validation Loss")
        ax[0, 0].set_title("Loss over Epochs")
        ax[0, 0].set_xlabel("Epoch")
        ax[0, 0].set_ylabel("Loss")
        ax[0, 0].legend()
        ax[0, 0].grid()

        ax[0, 1].plot(epochs, metrics["accuracy"], "b-", label="Training Accuracy")
        ax[0, 1].plot(
            epochs, metrics["val_accuracy"], "r-", label="Validation Accuracy"
        )
        ax[0, 1].set_title("Accuracy over Epochs")
        ax[0, 1].set_xlabel("Epoch")
        ax[0, 1].set_ylabel("Accuracy")
        ax[0, 1].set_ylim(0, 1)
        ax[0, 1].legend()
        ax[0, 1].grid()

        im = ax[1, 0].imshow(metrics["confusion_matrix"], cmap="Blues")
        ax[1, 0].set_title("Confusion Matrix")
        ax[1, 0].set_xlabel("Predicted Label")
        ax[1, 0].set_ylabel("True Label")
        fig.colorbar(im, ax=ax[1, 0])

        ax[1, 1].axis("off")
    else:
        # Test metrics (single values)
        fig, ax = plt.subplots(1, 2, figsize=(12, 5))

        ax[0].bar(["Accuracy"], [metrics["accuracy"]], color="blue")
        ax[0].set_ylim(0, 1)
        ax[0].set_title("Test Accuracy")
        ax[0].set_ylabel("Accuracy")

        ax[1].bar(["Loss"], [metrics["loss"]], color="red")
        ax[1].set_title("Test Loss")
        ax[1].set_ylabel("Loss")

    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()


def evaluate_own_photo(model, image_path):
    from PIL import Image
    import numpy as np

    img = Image.open(image_path).convert("L")  # Convert to grayscale
    img = img.resize((28, 28))  # Resize to 28x28
    img_array = np.array(img) / 255.0  # Normalize to [0, 1]
    img_array = (
        1.0 - img_array
    )  # Invert colors from white background to black background
    img_array = img_array.reshape(1, 28, 28, 1)  # Reshape for model input

    # Make prediction
    predictions = model.predict(img_array)
    predicted_label = np.argmax(predictions, axis=1)[0]
    labels = {
        0: "T-shirt/top",
        1: "Trouser",
        2: "Pullover",
        3: "Dress",
        4: "Coat",
        5: "Sandal",
        6: "Shirt",
        7: "Sneaker",
        8: "Bag",
        9: "Ankle boot",
    }
    confidence = np.max(predictions)

    print(f"Predicted Label: {labels[predicted_label]}, Confidence: {confidence:.4f}")


if __name__ == "__main__":
    plot_both_types(save_path="plots/training_metrics_both.png")
