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

    def build(self, input_shape):
        super(Encoder, self).build(input_shape)

        self.conv1.build(input_shape)
        self.bn1.build(self.conv1.compute_output_shape(input_shape))

        shape_after_conv1 = self.conv1.compute_output_shape(input_shape)
        self.conv2.build(shape_after_conv1)
        self.bn2.build(self.conv2.compute_output_shape(shape_after_conv1))

        shape_after_conv2 = self.conv2.compute_output_shape(shape_after_conv1)
        self.conv3.build(shape_after_conv2)
        self.bn3.build(self.conv3.compute_output_shape(shape_after_conv2))

        shape_after_conv3 = self.conv3.compute_output_shape(shape_after_conv2)
        self.flatten.build(shape_after_conv3)
        self.dense.build(self.flatten.compute_output_shape(shape_after_conv3))

        self.built = True

    def call(self, x, training=None):
        x = self.conv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.conv3(x)
        x = self.bn3(x, training=training)
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

    def build(self, input_shape): 
        super(Decoder, self).build(input_shape)

        # Build layers sequentially with proper shape propagation
        self.dense.build(input_shape)
        shape_after_dense = self.dense.compute_output_shape(input_shape)
        self.reshape.build(shape_after_dense)

        shape_after_reshape = self.reshape.compute_output_shape(shape_after_dense)
        self.deconv1.build(shape_after_reshape)
        self.bn1.build(self.deconv1.compute_output_shape(shape_after_reshape))

        shape_after_deconv1 = self.deconv1.compute_output_shape(shape_after_reshape)
        self.deconv2.build(shape_after_deconv1)
        self.bn2.build(self.deconv2.compute_output_shape(shape_after_deconv1))

        shape_after_deconv2 = self.deconv2.compute_output_shape(shape_after_deconv1)
        self.deconv3.build(shape_after_deconv2)
        self.bn3.build(self.deconv3.compute_output_shape(shape_after_deconv2))

        shape_after_deconv3 = self.deconv3.compute_output_shape(shape_after_deconv2)
        self.output_layer.build(shape_after_deconv3)

        self.built = True

    def call(self, x, training=None):
        x = self.dense(x)
        x = self.reshape(x)

        x = self.deconv1(x)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)

        x = self.deconv2(x)
        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)

        x = self.deconv3(x)
        x = self.bn3(x, training=training)
        x = tf.nn.relu(x)

        x = self.output_layer(x)

        return x


class Autoencoder(Model):
    def __init__(self, input_dim, latent_dim):
        super(Autoencoder, self).__init__()
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.encoder = Encoder(input_dim, latent_dim)
        self.decoder = Decoder(latent_dim, input_dim)

    def build(self, input_shape):
        super(Autoencoder, self).build(input_shape)

        # Build encoder with input shape
        self.encoder.build(input_shape)

        # Build decoder with latent shape
        latent_shape = (input_shape[0], self.latent_dim)
        self.decoder.build(latent_shape)

        self.built = True # Mark the model as built

    def call(self, x, training=None):
        z = self.encoder(x, training=training)
        reconstructed = self.decoder(z, training=training)
        return reconstructed
