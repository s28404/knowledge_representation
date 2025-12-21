import tensorflow as tf
from tensorflow import keras


def train_transform(image):
    # resize from (32, 32, 3) to (40, 40, 3) and then randomly crop back to (32, 32, 3)
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    # From [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    # From [0, 1] to [-1, 1] because the autoencoder uses tanh activation
    image = (image - 0.5) / 0.5
    return image



def load_data(batch_size):
    # Load CIFAR-10 using keras.datasets
    (x_train, _), (x_test, _) = keras.datasets.cifar10.load_data()

    train_loader = (
        # from_tensor_slices creates a dataset from the given tensors
        tf.data.Dataset.from_tensor_slices((x_train, x_train))
        # Apply the train_transform to both input and target 
        .map(lambda x, y: (train_transform(x), train_transform(y)), num_parallel_calls=tf.data.AUTOTUNE)
        # buffer_size=10000 uses a buffer of 10,000 samples for shuffling but finally all samples are shuffled
        .shuffle(buffer_size=10000)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    return train_loader
