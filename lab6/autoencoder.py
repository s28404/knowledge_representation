import tensorflow as tf
from keras import layers, Sequential


def create_encoder(input_dim, latent_dim):
    model = Sequential([
        layers.Conv2D(32, kernel_size=3, strides=2, padding="same", input_shape=input_dim),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2D(64, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2D(128, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Flatten(),
        layers.Dense(latent_dim),
    ])
    return model


def create_decoder(latent_dim):
    model = Sequential([
        layers.Dense(4 * 4 * 128, input_shape=(latent_dim,)),
        layers.Reshape((4, 4, 128)),
        
        layers.Conv2DTranspose(128, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(64, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2DTranspose(32, kernel_size=3, strides=2, padding="same"),
        layers.BatchNormalization(),
        layers.ReLU(),
        
        layers.Conv2D(3, kernel_size=3, padding="same", activation="tanh"),
    ])
    return model



class Autoencoder(tf.keras.Model):
    def __init__(self, input_dim, latent_dim):
        super().__init__()
        self.encoder = create_encoder(input_dim, latent_dim)
        self.decoder = create_decoder(latent_dim)

    def call(self, x):
        z = self.encoder(x)
        reconstructed = self.decoder(z)
        return reconstructed
