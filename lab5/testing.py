import json
import tensorflow as tf
import os


class Tester:
    def __init__(self, compiled_model, test_dataset, model_type):
        self.model = compiled_model
        self.test_dataset = test_dataset
        self.model_type = model_type

    def evaluate(self):
        test_loss, test_acc = self.model.evaluate(self.test_dataset)
        print(f"Test Accuracy: {test_acc:.4f}, Test Loss: {test_loss:.4f}")
        confusion_matrix = tf.math.confusion_matrix(
            # from (batch_size, ) tensors to single (num_samples, ) tensor, axis=0 to concatenate along samples
            tf.concat([y for x, y in self.test_dataset], axis=0),
            # from (num_samples, num_classes) to (num_samples, ) by taking the index of the max logit
            # where logit is the predicted class before applying softmax
            tf.argmax(self.model.predict(self.test_dataset), axis=1),
        )
        print("Confusion Matrix:")
        # from tensor to numpy array for matplotlib or printing
        print(confusion_matrix.numpy())
        os.makedirs("metrics", exist_ok=True)
        json.dump(
            {
                "loss": test_loss,
                "accuracy": test_acc,
                "confusion_matrix": confusion_matrix.numpy().tolist(),
            },
            open(f"metrics/test_metrics_{self.model_type}.json", "w"),
        )
