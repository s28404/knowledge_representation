import keras


def model_dense(hp):
    model = keras.Sequential()
    model.add(keras.layers.Flatten(input_shape=(28, 28, 1)))

    for i in range(hp.Int("num_layers", 1, 3)):
        model.add(
            # from 32 to 512 units, add 32 units each step were each step is already a separate model
            keras.layers.Dense(
                units=hp.Int(f"units_{i}", min_value=32, max_value=512, step=32),
                activation="relu",
            )
        )

    # we want probabilities for 10 classes
    model.add(keras.layers.Dense(10, activation="softmax"))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        # sparce categorical crossentropy since labels are integers
        # categorical crossentropy would be used if labels were one-hot encoded
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model


def model_cnn(hp):
    model = keras.Sequential()
    model.add(keras.layers.Input(shape=(28, 28, 1)))

    for i in range(hp.Int("num_conv_layers", 1, 3)):
        model.add(
            keras.layers.Conv2D(
                # from 32 to 256 filters, add 32 filters each step, were each step is already a separate model
                filters=hp.Int(f"filters_{i}", min_value=32, max_value=256, step=32),
                kernel_size=(3, 3),
                activation="relu",
            )
        )
        model.add(keras.layers.MaxPooling2D(pool_size=(2, 2)))

    # from 2D to 1D for Dense layer
    model.add(keras.layers.Flatten())

    model.add(
        keras.layers.Dense(
            units=hp.Int("dense_units", min_value=32, max_value=256, step=32),
            activation="relu",
        )
    )

    model.add(keras.layers.Dense(10, activation="softmax"))

    hp_learning_rate = hp.Choice("learning_rate", values=[1e-2, 1e-3, 1e-4])

    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=hp_learning_rate),
        # sparse categorical crossentropy since labels are integers
        # categorical crossentropy would be used if labels were one-hot encoded
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    return model
