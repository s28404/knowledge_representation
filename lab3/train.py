import keras
from keras.optimizers import Adam
import datetime
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import os


def model_v1():
    model = keras.Sequential()
    model.add(
        keras.layers.Dense(32, activation="relu", kernel_initializer="he_uniform", name="hidden_layer_1")
    )
    model.add(keras.layers.Dense(64, activation="relu", name="hidden_layer_2"))
    model.add(keras.layers.Dense(3, activation="softmax", name="output_layer"))
    return model


def model_v2():
    model = keras.Sequential()
    model.add(keras.layers.Dense(16, activation="elu", name="hidden_layer_1"))
    model.add(keras.layers.Dropout(0.3))
    model.add(keras.layers.Dense(32, activation="elu", name="hidden_layer_2"))
    model.add(keras.layers.Dropout(0.3))
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
    plt.legend()

    plt.tight_layout()
    plt.savefig(f"plots/training_history_{model_type}.png")


def train_models(x_train, x_test, y_train, y_test, epochs, batch_size):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=f"logs/{timestamp}")

    m1 = model_v1()
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

    m2 = model_v2()
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

    print(f"\nV1: accuracy={acc1:.4f}, Loss: {loss1:.4f}")
    print(f"V2: accuracy={acc2:.4f}, Loss: {loss2:.4f}")

    m1.save("model_v1.keras")
    m2.save("model_v2.keras")
    print("Models saved")
