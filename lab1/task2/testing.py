import tensorflow as tf
import wandb
import numpy as np

class Tester:
    def __init__(self, model, test_loader, loss_fn, seed):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn
        self.seed = seed

    @tf.function
    def test_step(self, x):
        reconstructed = self.model(x, training=False)
        loss = self.loss_fn(x, reconstructed)
        return {"loss": loss, "reconstructed": reconstructed}

    def test_batch(self, batch):
        x = batch
        step_metric = self.test_step(x)
        return {
            "loss": step_metric["loss"],
            "reconstructed": step_metric["reconstructed"],
        }

    def test(self):
        test_stats = {"losses": [], "avg_loss": 0.0}

        photos = {"original": [], "reconstructed": []}

        for batch in self.test_loader:
            metrics = self.test_batch(batch)
            test_stats["losses"].append(metrics["loss"].numpy())
            photos["original"].append(batch.numpy())
            photos["reconstructed"].append(metrics["reconstructed"].numpy())
            wandb.log({"test_loss": metrics["loss"].numpy()})

        avg_loss = sum(test_stats["losses"]) / len(test_stats["losses"])
        test_stats["avg_loss"] = avg_loss

        print(f"Test Loss: {avg_loss:.4f}, Seed: {self.seed}")

        wandb.log({"average_test_loss": avg_loss})

        # Concatenate batches into single arrays for plotting
        photos["original"] = np.concatenate(photos["original"], axis=0)
        photos["reconstructed"] = np.concatenate(photos["reconstructed"], axis=0)

        return test_stats, photos
