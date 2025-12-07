import argparse
from dataset import prepare_dataset, load_fashion_mnist
from models import model_dense, model_cnn
from tools import load_hyperparameters, plot, evaluate_own_photo, plot_both_types
from training import Trainer
from testing import Tester
import os
import tensorflow as tf


def main():
    parser = argparse.ArgumentParser(description="Train a model on Fashion MNIST")
    parser.add_argument(
        "--model_type",
        type=str,
        choices=["dense", "cnn"],
        default="dense",
        help="Type of model to train: 'dense' or 'cnn'",
    )
    parser.add_argument(
        "--own_photo",
        type=str,
        default=None,
        help="Path to an own photo for evaluation",
    )
    args = parser.parse_args()

    if args.own_photo:
        # On my local laptop I uses tensorflow 2.21.0-dev20250925 with which I saved the model
        # but on the server tensorflow 2.12.0 is installed, so I need to load the model differently
        model = tf.keras.models.load_model(
            f"models/{args.model_type}_fashion_mnist_model.h5"
        )
        evaluate_own_photo(model, args.own_photo)
        return

    hyperparams = load_hyperparameters()

    train_dataset, test_dataset = load_fashion_mnist()
    train_dataset = prepare_dataset(train_dataset, batch_size=hyperparams["batch_size"])
    test_dataset = prepare_dataset(test_dataset, batch_size=hyperparams["batch_size"])

    if args.model_type == "dense":
        model_builder = model_dense
    else:
        model_builder = model_cnn

    trainer = Trainer(
        model_builder=model_builder,
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        epochs=hyperparams["epochs"],
        model_type=args.model_type,
    )

    trainer.tune()

    os.makedirs("models", exist_ok=True)
    trainer.save_model(f"models/{args.model_type}_fashion_mnist_model.keras")
    trainer.train()

    os.makedirs("plots", exist_ok=True)

    plot(
        parameters=f"metrics/train_metrics_{args.model_type}.json",
        save_path=f"plots/training_metrics_{args.model_type}.png",
    )

    tester = Tester(
        compiled_model=trainer.model,
        test_dataset=test_dataset,
        model_type=args.model_type,
    )
    tester.evaluate()
    plot(
        parameters=f"metrics/test_metrics_{args.model_type}.json",
        save_path=f"plots/testing_metrics_{args.model_type}.png",
    )

    plot_both_types(save_path="plots/training_metrics_both.png")


if __name__ == "__main__":
    main()
