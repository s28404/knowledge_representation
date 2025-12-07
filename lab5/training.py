import tensorflow as tf
import json
import keras_tuner as kt
import os


class MetricsCallback(tf.keras.callbacks.Callback):
    def __init__(self):
        self.losses = []
        self.accuracies = []
        self.val_losses = []
        self.val_accuracies = []

    def on_epoch_end(self, epoch, logs=None):
        self.losses.append(float(logs["loss"]))
        self.accuracies.append(float(logs["accuracy"]))
        self.val_losses.append(float(logs["val_loss"]))
        self.val_accuracies.append(float(logs["val_accuracy"]))


class Trainer:
    def __init__(
        self, model_builder, train_dataset, test_dataset, epochs=10, model_type="dense"
    ):
        self.model_builder = model_builder
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset
        self.epochs = epochs
        self.model_type = model_type
        self.model = None
        self.tuner = None
        self.metrics_callback = None

    def tune(self):
        self.tuner = kt.Hyperband(
            hypermodel=self.model_builder,
            objective="val_accuracy",
            max_epochs=4,
            directory="kt_dir",
            project_name=f"fashion_mnist_{self.model_type}",
        )

        print(f"Starting hyperparameter tuning for {self.model_type} model...")
        self.tuner.search(
            self.train_dataset,
            epochs=4,
            validation_data=self.test_dataset,
            verbose=1,
        )

        best_hp = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        print("Best hyperparameters:")
        for param in best_hp.values:
            print(f"  {param}: {best_hp.get(param)}")

        print(f"\nRetraining best model with {self.epochs} epochs...")
        self.model = self.model_builder(best_hp)
        self.metrics_callback = MetricsCallback()
        self.model.fit(
            self.train_dataset,
            epochs=self.epochs,
            validation_data=self.test_dataset,
            callbacks=[self.metrics_callback],
            verbose=1,
        )
        print("Training completed")

    def train(self):
        confusion_matrix = tf.math.confusion_matrix(
            tf.concat([y for x, y in self.train_dataset], axis=0),
            tf.argmax(self.model.predict(self.train_dataset, verbose=0), axis=1),
        )
        print("\nConfusion Matrix:")
        print(confusion_matrix.numpy())

        os.makedirs("metrics", exist_ok=True)
        json.dump(
            {
                "loss": self.metrics_callback.losses,
                "accuracy": self.metrics_callback.accuracies,
                "val_loss": self.metrics_callback.val_losses,
                "val_accuracy": self.metrics_callback.val_accuracies,
                "confusion_matrix": confusion_matrix.numpy().tolist(),
            },
            open(f"metrics/train_metrics_{self.model_type}.json", "w"),
        )

    def save_model(self, filepath):
        self.model.save(filepath)
