import tensorflow as tf
from keras import layers, Model


class Encoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Encoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim

        self.conv1 = layers.Conv2D(
            32, kernel_size=3, strides=2, padding="same"
        )  # (32x32x3) -> (16x16x32)
        self.bn1 = layers.BatchNormalization()

        self.conv2 = layers.Conv2D(
            64, kernel_size=3, strides=2, padding="same"
        )  # (16x16x32) -> (8x8x64)
        self.bn2 = layers.BatchNormalization()

        self.conv3 = layers.Conv2D(
            128, kernel_size=3, strides=2, padding="same"
        )  # (8x8x64) -> (4x4x128)
        self.bn3 = layers.BatchNormalization()

        self.flatten = layers.Flatten()  # (4x4x128) -> (2048)
        self.dense = layers.Dense(latent_dim)  # (2048) -> (latent_dim)

    def call(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)

        x = self.flatten(x)
        x = self.dense(x)

        return x


class Decoder(Model):
    def __init__(self, latent_dim, output_dim):
        super(Decoder, self).__init__()
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.dense = layers.Dense(4 * 4 * 128)  # (latent_dim) -> (2048)
        self.reshape = layers.Reshape((4, 4, 128))  # (2048) -> (4x4x128)

        self.deconv1 = layers.Conv2DTranspose(
            128, kernel_size=3, strides=2, padding="same"
        )  # (4x4x128) -> (8x8x128)
        self.bn1 = layers.BatchNormalization()

        self.deconv2 = layers.Conv2DTranspose(
            64, kernel_size=3, strides=2, padding="same"
        )  # (8x8x128) -> (16x16x64)
        self.bn2 = layers.BatchNormalization()

        self.deconv3 = layers.Conv2DTranspose(
            32, kernel_size=3, strides=2, padding="same"
        )  # (16x16x64) -> (32x32x32)
        self.bn3 = layers.BatchNormalization()

        self.output_layer = layers.Conv2D(
            3, kernel_size=3, padding="same", activation="tanh"
        )  # (32x32x32) -> (32x32x3)

    def call(self, x):
        x = self.dense(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x)
        x = tf.nn.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x)
        x = tf.nn.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x)
        x = tf.nn.relu(x)

        x = self.output_layer(x)

        return x


class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
