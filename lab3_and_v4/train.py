import keras
from keras.optimizers import Adam
import datetime
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os
import keras_tuner as kt
from sklearn.metrics import confusion_matrix
import numpy as np


def build_model_v1(units1=64, units2=64, activation="relu"):
    normalizater = keras.layers.Normalization(input_shape=(13,))
    model = keras.Sequential([normalizater])
    model.add(
        keras.layers.Dense(
            units1,
            activation=activation,
            kernel_initializer="he_uniform",
            name="hidden_layer_1",
        )
    )
    model.add(keras.layers.Dense(units2, activation=activation, name="hidden_layer_2"))
    model.add(keras.layers.Dense(3, activation="softmax", name="output_layer"))
    return model


def build_model_v2(units1=32, units2=32, activation="relu", dropout=0.2):
    normalizater = keras.layers.Normalization(input_shape=(13,))
    model = keras.Sequential([normalizater])
    model.add(keras.layers.Dense(units1, activation=activation, name="hidden_layer_1"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(units2, activation=activation, name="hidden_layer_2"))
    model.add(keras.layers.Dropout(dropout))
    model.add(keras.layers.Dense(3, activation="softmax", name="output_layer"))
    return model


def model_v1(hp):
    normalizater = keras.layers.Normalization(input_shape=(13,))
    # We add [normalizater] to the Sequential model to ensure the input goes through normalization first
    model = keras.Sequential([normalizater])
    model.add(
        keras.layers.Dense(
            hp.Choice("units1", [32, 64, 128]),
            activation=hp.Choice("activation", ["relu", "tanh", "elu"]),
            kernel_initializer="he_uniform",
            name="hidden_layer_1",
        )
    )
    model.add(
        keras.layers.Dense(
            hp.Choice("units2", [32, 64, 128]),
            hp.Choice("activation", ["relu", "tanh", "elu"]),
            name="hidden_layer_2",
        )
    )
    model.add(keras.layers.Dense(3, activation="softmax", name="output_layer"))
    return model


def model_v2(hp):
    normalizater = keras.layers.Normalization(input_shape=(13,))
    model = keras.Sequential([normalizater])
    model.add(
        keras.layers.Dense(
            hp.Choice("units1", [16, 32, 64]),
            activation=hp.Choice("activation", ["relu", "tanh", "elu"]),
            name="hidden_layer_1",
        )
    )
    model.add(keras.layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(
        keras.layers.Dense(
            hp.Choice("units2", [16, 32, 64]),
            hp.Choice("activation", ["relu", "tanh", "elu"]),
            name="hidden_layer_2",
        )
    )
    model.add(keras.layers.Dropout(hp.Float("dropout", 0.1, 0.5, step=0.1)))
    model.add(keras.layers.Dense(3, activation="softmax", name="output_layer"))
    return model


def plot(history, model_type="v2"):
    os.makedirs("plots", exist_ok=True)

    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(history["accuracy"], label=f"{model_type} Train Acc")
    plt.plot(history["val_accuracy"], label=f"{model_type} Val Acc")
    plt.title(f"Model {model_type} Accuracy")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(history["loss"], label=f"{model_type} Train Loss")
    plt.plot(history["val_loss"], label=f"{model_type} Val Loss")
    plt.title(f"Model {model_type} Loss")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")


def tune_model_v2(
    x_train, x_test, y_train, y_test, max_trials=20, executions_per_trial=1
):
    def build_model(hp):
        model = model_v2(hp)
        # adapt makes the normalization layer learn mean and std from training data
        model.layers[0].adapt(x_train)
        # sampling="log" means logarithmic scale from 1e-4 to 1e-2
        learning_rate = hp.Float(
            "learning_rate", min_value=1e-4, max_value=1e-2, sampling="log"
        )
        model.compile(
            optimizer=Adam(learning_rate),
            loss="categorical_crossentropy",
            metrics=["accuracy"],
        )
        return model

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=max_trials,
        # executions_per_trial is the number of times to train the model with the same hyperparameters
        executions_per_trial=executions_per_trial,
        directory="tuner_logs",
        project_name="wine_v2",
    )

    tuner.search(
        x_train,
        y_train,
        epochs=10,
        validation_data=(x_test, y_test),
        callbacks=[
            keras.callbacks.TensorBoard(
                log_dir=f"logs/tune_{datetime.datetime.now().strftime('%H%M%S')}"
            )
        ],
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    print(f"Best hyperparameters: {best_hps.values}")

    # num_models = 1 to get the best model, tuner.get_best_models returns a list of models, so we take the first element (best model)
    best_model = tuner.get_best_models(num_models=1)[0]
    loss, acc = best_model.evaluate(x_test, y_test)
    print(f"Best model accuracy: {acc:.4f}")

    best_model.save("tuned_model_improved_v2.keras")
    print("Tuned model saved")

    return best_model, acc


def train_models(x_train, x_test, y_train, y_test, epochs, batch_size):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f"logs/{timestamp}")

    m1 = build_model_v1()
    # layer[0] is the normalization layer
    m1.layers[0].adapt(x_train)  # Adapt normalization layer
    m1.compile(
        optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print("Training Model V1")
    h1 = m1.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[tensorboard_callback],
    )

    m2 = build_model_v2()
    m2.layers[0].adapt(x_train)  # Adapt normalization layer
    m2.compile(
        optimizer=Adam(0.001), loss="categorical_crossentropy", metrics=["accuracy"]
    )
    print("\nTraining Model V2")
    h2 = m2.fit(
        x_train,
        y_train,
        batch_size=batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        verbose=1,
        callbacks=[tensorboard_callback],
    )

    plot(h1.history, model_type="v1")
    plot(h2.history, model_type="v2")

    loss1, acc1 = m1.evaluate(x_test, y_test)
    loss2, acc2 = m2.evaluate(x_test, y_test)

    print(f"\nBaseline V1: accuracy={acc1:.4f}, Loss: {loss1:.4f}")
    print(f"Baseline V2: accuracy={acc2:.4f}, Loss: {loss2:.4f}")

    y_pred1 = m1.predict(x_test)
    y_pred_classes1 = np.argmax(y_pred1, axis=1)
    y_true = np.argmax(y_test, axis=1)
    cm1 = confusion_matrix(y_true, y_pred_classes1)

    print("Confusion Matrix V1:")
    print(cm1)

    y_pred2 = m2.predict(x_test)
    y_pred_classes2 = np.argmax(y_pred2, axis=1)
    cm2 = confusion_matrix(y_true, y_pred_classes2)
    print("Confusion Matrix V2:")
    print(cm2)

    m1.save("model_improved_v1.keras")
    m2.save("model_improved_v2.keras")
    print("Models saved")

    if acc1 > acc2:
        print(f"Model V1 is better (accuracy: {acc1:.4f} vs {acc2:.4f})")
    elif acc2 > acc1:
        print(f"Model V2 is better (accuracy: {acc2:.4f} vs {acc1:.4f})")
    else:
        print(f"Models have the same accuracy: {acc1:.4f}")
