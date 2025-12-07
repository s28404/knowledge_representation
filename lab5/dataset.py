import tensorflow as tf
import tensorflow_datasets as tfds
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use("Agg")


# with as_supervised=False the dataset returns a dictionary with 'image' and 'label' keys
# with as_supervised=True it returns a tuple (image, label)
def load_fashion_mnist(split=["train", "test"], as_supervised=True):
    train_ds, test_ds = tfds.load(
        "fashion_mnist", split=split, as_supervised=as_supervised
    )
    return train_ds, test_ds


def augment_image(image, label):
    # from [0, 255] to [0, 1]
    image = tf.cast(image, tf.float32) / 255.0
    image = tf.image.random_flip_left_right(image)
    image = tf.image.random_flip_up_down(image)
    # +- 20% contrast
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    # +- 10% brightness
    image = tf.image.random_brightness(image, max_delta=0.1)
    return image, label


def prepare_dataset(dataset, batch_size=32, shuffle_buffer_size=1000):
    # num_parallel_calls for performance optimization using multiple CPU cores
    dataset = dataset.map(augment_image, num_parallel_calls=tf.data.AUTOTUNE)
    dataset = dataset.shuffle(shuffle_buffer_size)
    dataset = dataset.batch(batch_size)
    # prefetch for performance optimization by overlapping data preprocessing and model execution
    dataset = dataset.prefetch(tf.data.AUTOTUNE)
    return dataset


if __name__ == "__main__":
    train_dataset, test_dataset = load_fashion_mnist()
    train_dataset = prepare_dataset(train_dataset, batch_size=64)

    for images, labels in train_dataset.take(1):
        plt.figure(figsize=(10, 10))
        for i in range(9):
            ax = plt.subplot(3, 3, i + 1)
            # from tensor to numpy array, squeeze to from (28, 28, 1) to (28, 28), matplotlib needs 2D array for grayscale
            plt.imshow(images[i].numpy().squeeze())
            # from tensor to numpy scalar
            plt.title(int(labels[i].numpy()))
            plt.axis("off")
        plt.savefig("fashion_mnist_sample.png")
