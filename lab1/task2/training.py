import tensorflow as tf
import wandb


class Trainer:
    def __init__(
        self,
        model,
        train_loader,
        loss_fn,
        optimizer,
        checkpoint,
        checkpoint_interval,
        checkpoint_dir,
        recon_log_interval,
        seed,
    ):
        self.model = model
        self.train_loader = train_loader
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        self.checkpoint = checkpoint
        self.checkpoint_interval = checkpoint_interval
        self.checkpoint_dir = checkpoint_dir
        self.recon_log_interval = recon_log_interval
        self.seed = seed


    @tf.function
    def train_step(self, x):
        with tf.GradientTape() as tape:
            reconstructed = self.model(x, training=True)
            loss = self.loss_fn(x, reconstructed)
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        return {"loss": loss}

    @tf.function
    def train_batch(self):
        batch_stats = {"avg_loss": 0.0}

        for i, batch in enumerate(self.train_loader):
            real_images = batch
            loss = self.train_step(real_images)
            batch_stats["avg_loss"] += loss["loss"] / tf.cast(
                i + 1, tf.float32
            )  # cast to float to avoid int division

        return batch_stats

    def train(self, epochs):
        train_stats = {"avg_losses": []}

        for epoch in range(epochs):
            metrics = self.train_batch()
            train_stats["avg_losses"].append(metrics["avg_loss"].numpy())
            
            lr_attr = self.optimizer.learning_rate # EagerTensor
            current_lr = float(tf.convert_to_tensor(lr_attr))
            
            print(
                f"Epoch: {epoch + 1}, Avg Loss: {metrics['avg_loss'].numpy():.4f}, LR: {current_lr:.6f}, Seed: {self.seed}"
            )
            wandb.log(
                {
                    "epoch": epoch + 1,
                    "avg_loss": metrics["avg_loss"].numpy(),
                    "learning_rate": current_lr,
                }
            )

            if self.checkpoint and (epoch + 1) % self.checkpoint_interval == 0:
                self.checkpoint.save()

            if (epoch + 1) % self.recon_log_interval == 0:
                sample_batch = next(iter(self.train_loader))
                reconstructed = self.model(sample_batch, training=False)
                comparison = tf.concat([sample_batch[:8], reconstructed[:8]], axis=0)
                comparison = (comparison + 1) / 2.0  # [-1, 1] -> [0, 1]
                wandb.log(
                    {
                        "reconstructions": [
                            wandb.Image(img) for img in comparison.numpy()
                        ]
                    }
                )

        return train_stats
