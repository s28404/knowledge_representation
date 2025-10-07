import argparse
import tensorflow as tf
from keras import optimizers
import wandb
import secrets
from data import load_data
from autoencoder import Autoencoder
from training import Trainer
from testing import Tester
from utils import (
    plot_loss,
    plot_recons,
    load_hyperparameters,
    save_stats,
    save_metadata,
)


def main():
    parser = argparse.ArgumentParser(description="Autoencoder Training and Testing")
    parser.add_argument(
        "--type", type=str, default="train", help="Type of run: train or test"
    )
    parser.add_argument(
        "--seed",
        type=str,
        default=None,
        help="Set own seed to for example load specific checkpoint",
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="[REQUIRED for testing] Path to checkpoint directory (e.g., checkpoints/seed_123/ckpt-20)",
    )
    args = parser.parse_args()

    config = load_hyperparameters("hyperparameters.yaml")

    if args.seed is not None:
        seed = int(args.seed)
    else:
        seed = secrets.randbelow(2**32)

    tf.random.set_seed(seed)

    wandb.init(project=config["project_name"], name=f"{args.type}_seed_{seed}")
    wandb.config.update({"seed": seed})
    wandb.config.update(config)

    train_loader, test_loader = load_data(config["batch_size"])

    model = Autoencoder(latent_dim=config["latent_dim"], input_dim=(32, 32, 3))
    model.build(input_shape=(None, 32, 32, 3))

    loss_fn = tf.keras.losses.MeanSquaredError()

    if args.type == "train":
        steps_per_epoch = len(train_loader)
        decay_steps = steps_per_epoch * config["epochs"]

        scheduler = optimizers.schedules.CosineDecay(
            initial_learning_rate=config["learning_rate"],
            decay_steps=decay_steps,
            alpha=config["alpha"],  # minimum learning rate
        )
        optimizer = optimizers.Adam(learning_rate=scheduler)

        metadata = save_metadata(model, config, seed)

        wandb.config.update(
            {
                "total_params": metadata["model_parameters"]["total_params"],
                "trainable_params": metadata["model_parameters"]["trainable_params"],
                "non_trainable_params": metadata["model_parameters"][
                    "non_trainable_params"
                ],
            }
        )

        checkpoint = tf.train.Checkpoint(optimizer=optimizer, model=model)
        checkpoint_manager = tf.train.CheckpointManager(
            checkpoint,
            directory=f"{config['checkpoint_dir']}/seed_{seed}",
            max_to_keep=config["max_to_keep"],  # how many latest checkpoints to keep
        )

        # Auto-resume from latest checkpoint if exists
        if checkpoint_manager.latest_checkpoint:
            status = checkpoint.restore(checkpoint_manager.latest_checkpoint)
            print(f"[TRAINING] Restored from {checkpoint_manager.latest_checkpoint}")
            status.expect_partial()  # Don't warn about optimizer state
        else:
            print("[TRAINING] Initializing from scratch.")

        trainer = Trainer(
            model=model,
            train_loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            checkpoint=checkpoint_manager,
            checkpoint_interval=config["checkpoint_interval"],
            checkpoint_dir=config["checkpoint_dir"],
            recon_log_interval=config["recon_log_interval"],
            seed=seed,
        )
        train_stats = trainer.train(epochs=config["epochs"])
        save_stats(
            train_stats, stats_dir="stats/train", file_name="train_stats", seed=seed
        )
        plot_loss(train_stats, file_name="training_loss", seed=seed)

    elif args.type == "test":
        if not args.checkpoint:
            raise ValueError(
                "Testing requires --checkpoint argument.\n"
                "Example: python main.py --type test --checkpoint checkpoints/seed_954871688/ckpt-20"
            )

        if not tf.io.gfile.exists(args.checkpoint + ".index"): # Check if metadata file exists
            raise ValueError(f"Error: {args.checkpoint} not found.")
        
        ckpt = tf.train.Checkpoint(model=model)
        ckpt.restore(args.checkpoint).expect_partial()


        tester = Tester(
            model=model, test_loader=test_loader, loss_fn=loss_fn, seed=seed
        )
        test_stats, reconstructions = tester.test()

        save_stats(
            test_stats, stats_dir="stats/test", file_name="test_stats", seed=seed
        )
        plot_recons(
            originals=reconstructions["original"],
            reconstructions=reconstructions["reconstructed"],
            file_name="test_reconstructions",
            seed=seed,
        )


if __name__ == "__main__":
    main()
