import argparse
import tensorflow as tf
from keras import optimizers
import secrets
from data import load_data
from autoencoder import Autoencoder
from training import Trainer
from utils import (
    plot_loss,
    load_hyperparameters,
    save_stats,
)


def main():
    parser = argparse.ArgumentParser(description="Autoencoder Training and Testing")
    parser.add_argument(
        "--type", type=str, default="train", help="Type of run: train or test"
    )
   
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")

 
    seed = secrets.randbelow(2**32)

    train_loader = load_data(config["batch_size"])

    model = Autoencoder(latent_dim=config["latent_dim"], input_dim=(32, 32, 3))

    loss_fn = tf.keras.losses.MeanSquaredError()

    scheduler = optimizers.schedules.CosineDecay(
            initial_learning_rate=config["learning_rate"],
            decay_steps=len(train_loader) * config["epochs"],
            # alpha is the minimum learning rate value
            alpha=config["alpha"],
        )
    optimizer = optimizers.Adam(learning_rate=scheduler)

    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
        )
    train_stats = trainer.train(epochs=config["epochs"])
    save_stats(
        train_stats, stats_dir="stats/train", file_name="train_stats", seed=seed
    )
    plot_loss(train_stats, file_name="training_loss", seed=seed)

if __name__ == "__main__":
    main()
