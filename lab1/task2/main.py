import argparse
import tensorflow as tf
from keras import optimizers
from data import load_data
from autoencoder import Autoencoder
from training import Trainer
from testing import Tester
import wandb
from utils import plot, load_hyperparameters, save_stats


def main():
    parser = argparse.ArgumentParser(description="Autoencoder Training and Testing")
    parser.add_argument(
        "--type", type=str, default="train", help="Type of run: train or test"
    )
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")
    train_loader, test_loader = load_data(config["batch_size"])

    steps_per_epoch = len(train_loader)
    decay_steps = steps_per_epoch * config["epochs"]

    scheduler = optimizers.schedules.CosineDecay(
        initial_learning_rate=config["learning_rate"],
        decay_steps=decay_steps,
        alpha=config["alpha"],  # minimum learning rate
    )
    optimizer = optimizers.Adam(learning_rate=scheduler)

    model = Autoencoder(latent_dim=config["latent_dim"], input_dim=(32, 32, 3))
    loss_fn = tf.keras.losses.MeanSquaredError()

    wandb.init(project=config["project_name"], name=args.type)
    if args.type == "train":
        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=config["checkpoint_dir"],
            max_to_keep=config["max_to_keep"],  # how many latest checkpoints to keep
        )
        if checkpoint_manager.latest_checkpoint:
            checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"Restored from {checkpoint_manager.latest_checkpoint}")
        else:
            print("Initializing from scratch.")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            checkpoint=checkpoint_manager,
            checkpoint_interval=config["checkpoint_interval"],
            checkpoint_dir=config["checkpoint_dir"],
            recon_log_interval=config["recon_log_interval"],
        )
        train_stats = trainer.train(epochs=config["epochs"])
        save_stats(train_stats, stats_dir="stats/train", file_name="train_stats.json")
        plot(train_stats, training=True, file_name="training_loss.png")

    elif args.type == "test":
        tester = Tester(model=model, test_loader=test_loader, loss_fn=loss_fn)
        test_stats, reconstructions = tester.test()
        save_stats(test_stats, stats_dir="stats/test", file_name="test_stats.json")
        plot(test_stats, training=False, file_name="testing_loss.png")


if __name__ == "__main__":
    main()
