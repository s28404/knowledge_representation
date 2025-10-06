import tensorflow as tf
import wandb

class Tester:
    def __init__(self, model, test_loader, loss_fn):
        self.model = model
        self.test_loader = test_loader
        self.loss_fn = loss_fn

    @tf.function
    def test_step(self, x):
        reconstructed = self.model(x, training=False)
        loss = self.loss_fn(x, reconstructed)
        return {"loss": loss, "reconstructed": reconstructed}
    
    def test_batch(self, batch):
        x = batch
        step_metric = self.test_step(x)
        return {"loss": step_metric["loss"], "reconstructed": step_metric["reconstructed"]}

    def test(self):
        test_stats = {"loss": []}
        all_reconstructions = []

        for batch in self.test_loader:
            metrics = self.test_batch(batch)
            test_stats["loss"].append(metrics["loss"].numpy())
            all_reconstructions.append(metrics["reconstructed"].numpy())

        avg_loss = sum(test_stats["loss"]) / len(test_stats["loss"])
        print(f"Test Loss: {avg_loss:.4f}")
        wandb.log({"test_loss": avg_loss})

        return test_stats, all_reconstructions