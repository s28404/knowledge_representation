import tensorflow as tf
import pandas as pd
import numpy as np


def train_transform(image):
    image = tf.image.resize_with_crop_or_pad(image, 40, 40)
    image = tf.image.random_crop(image, size=[32, 32, 3])
    image = tf.image.random_flip_left_right(image)

    image = tf.image.random_brightness(image, max_delta=0.2)
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)
    image = tf.image.random_saturation(image, lower=0.8, upper=1.2)
    image = tf.image.random_hue(image, max_delta=0.1)

    image = tf.cast(image, tf.float32) / 255.0  # Normalize to [0, 1]
    # tf.constant to explicitly work on tensorflow graph
    mean = tf.constant([0.5, 0.5, 0.5])
    std = tf.constant([0.5, 0.5, 0.5])
    image = (image - mean) / std  # Normalize to [-1, 1]
    return image


def test_transform(image):
    image = tf.cast(image, tf.float32) / 255.0
    mean = tf.constant([0.5, 0.5, 0.5])
    std = tf.constant([0.5, 0.5, 0.5])
    image = (image - mean) / std
    return image


def decode_image_dict(image_dict):
    img_bytes = image_dict["bytes"] # [32*32*3]
    img = tf.image.decode_png(img_bytes, channels=3)  # [32, 32, 3]
    return img

def load_data(batch_size):
    splits = {
        "train": "plain_text/train-00000-of-00001.parquet",
        "test": "plain_text/test-00000-of-00001.parquet",
    }
    train_set = pd.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["train"])
    test_set = pd.read_parquet("hf://datasets/uoft-cs/cifar10/" + splits["test"])
 
    # numpy stack to convert list of arrays to single array
    x_train = np.stack(train_set["img"].apply(decode_image_dict))
    x_test = np.stack(test_set["img"].apply(decode_image_dict))

    train_loader = (
        tf.data.Dataset.from_tensor_slices(x_train)  # (50000, 32, 32, 3)
        .map(
            lambda x: train_transform(x), num_parallel_calls=tf.data.AUTOTUNE
        )  # data augmentation per image
        .shuffle(buffer_size=10000)  # shuffle per 10000 images
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE)
    )

    test_loader = (
        tf.data.Dataset.from_tensor_slices(x_test)  # (10000, 32, 32, 3)
        .map(lambda x: test_transform(x), num_parallel_calls=tf.data.AUTOTUNE) # automatically determines optimal number of parallel threads (like pytorch num_workers)
        .batch(batch_size=batch_size)
        .prefetch(buffer_size=tf.data.AUTOTUNE) # overlaps data loading with training (like pytorch pin_memory=True)
    )

    return train_loader, test_loader

