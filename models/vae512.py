"""
https://keras.io/examples/generative/vae/
"""

import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.initializers import RandomNormal

from datetime import datetime

tf.config.run_functions_eagerly(True)

try:
    physical_devices = tf.config.list_physical_devices('GPU')
    tf.config.experimental.set_memory_growth(physical_devices[0], enable=True)
except:
    print('0')


class Sampling(layers.Layer):
    """Uses (z_mean, z_log_var) to sample z, the vector encoding a digit."""

    def call(self, inputs):
        z_mean, z_log_var = inputs
        batch = tf.shape(z_mean)[0]
        dim = tf.shape(z_mean)[1]
        epsilon = tf.keras.backend.random_normal(shape=(batch, dim))
        return z_mean + tf.exp(0.5 * z_log_var) * epsilon


class VAE(keras.Model):
    def __init__(self, encoder, decoder, **kwargs):
        super(VAE, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder
        self.total_loss_tracker = keras.metrics.Mean(name="total_loss")
        self.reconstruction_loss_tracker = keras.metrics.Mean(
            name="reconstruction_loss"
        )
        self.kl_loss_tracker = keras.metrics.Mean(name="kl_loss")

    @property
    def metrics(self):
        return [
            self.total_loss_tracker,
            self.reconstruction_loss_tracker,
            self.kl_loss_tracker,
        ]

    # @tf.function
    def train_step(self, data):
        data = [data[0]['input_1'], data[0]['input_2']]
        # if type(xcat_data) is dict:
        #     xcat_data = [xcat_data[0]['input_1'], xcat_data[0]['input_2']]
        with tf.GradientTape() as tape:
            # xcat_data = tf.convert_to_tensor(xcat_data)
            z_mean, z_log_var, z = self.encoder(data)
            reconstruction = self.decoder(z)
            reconstruction_loss = tf.reduce_mean(
                tf.reduce_sum(
                    keras.losses.binary_crossentropy(data, reconstruction), axis=(1, 2)
                )
            )
            kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
            kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
            total_loss = reconstruction_loss + kl_loss
        grads = tape.gradient(total_loss, self.trainable_weights)
        self.optimizer.apply_gradients(zip(grads, self.trainable_weights))
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return {
            "total_loss": self.total_loss_tracker.result(),
            "reconstruction_loss": self.reconstruction_loss_tracker.result(),
            "kl_loss": self.kl_loss_tracker.result(),
        }

    # implement the call method
    def call(self, inputs, *args, **kwargs):
        if type(inputs) is dict:
            inputs = [inputs['input_1'], inputs['input_2']]
        z_mean, z_log_var, z = self.encoder(inputs)
        reconstruction = self.decoder(z)
        reconstruction_loss = tf.reduce_mean(
            tf.reduce_sum(
                keras.losses.binary_crossentropy(inputs, reconstruction), axis=(1, 2)
            )
        )
        kl_loss = -0.5 * (1 + z_log_var - tf.square(z_mean) - tf.exp(z_log_var))
        kl_loss = tf.reduce_mean(tf.reduce_sum(kl_loss, axis=1))
        total_loss = reconstruction_loss + kl_loss
        self.total_loss_tracker.update_state(total_loss)
        self.reconstruction_loss_tracker.update_state(reconstruction_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        return reconstruction


def build_dual_encoder(input_shape=(32, 32, 1), inter_dim=16, latent_dim=32, filters=16, layers_n=3, merge_type='add'):
    e1_input = keras.Input(shape=input_shape)
    e2_input = keras.Input(shape=input_shape)

    e1_filters = filters
    e2_filters = filters

    inputs = [e1_input, e2_input]

    x1 = e1_input
    for i in range(layers_n):
        x1 = layers.Conv2D(e1_filters, 3, activation="relu", strides=2, padding="same")(x1)
        e1_filters *= 2

    x2 = e2_input
    for i in range(layers_n):
        x2 = layers.Conv2D(e2_filters, 3, activation="relu", strides=2, padding="same")(x2)
        e2_filters *= 2

    if merge_type == 'add':
        x = layers.Flatten()(layers.Add()([x1, x2]))
    else:
        x = layers.Flatten()(layers.Concatenate()([x1, x2]))

    x = layers.Dense(inter_dim, activation="relu")(x)
    z_mean = layers.Dense(latent_dim, name="z_mean")(x)
    z_log_var = layers.Dense(latent_dim, name="z_log_var")(x)
    z = Sampling()([z_mean, z_log_var])
    encoder = keras.Model(inputs, [z_mean, z_log_var, z], name="encoder")

    return encoder


def build_dual_decoder(input_shape=(32, 32, 1), latent_dim=32, filters=16, layers_n=3):
    # build decoder
    latent_inputs = keras.Input(shape=(latent_dim,))
    n = input_shape[0]
    c = input_shape[2]

    d1_filters = filters
    d2_filters = filters

    div_ = 2 ** layers_n

    # build branch 1 of decoder
    x1 = layers.Dense(n // div_ * n // div_ * div_ * filters, activation="relu")(latent_inputs)
    x1 = layers.Reshape((n // div_, n // div_, div_ * filters))(x1)

    d1_filters = d1_filters * 2 ** layers_n
    for i in range(layers_n):
        x1 = layers.Conv2DTranspose(d1_filters, 3, activation="relu", strides=2, padding="same")(x1)
        d1_filters //= 2

    # d1_outputs = layers.Conv2DTranspose(c, 3, activation="sigmoid", strides=2, padding="same")(x1)
    d1_outputs = layers.Conv2DTranspose(c, 3, activation="sigmoid", padding="same")(x1)

    # build branch 2 of decoder
    x2 = layers.Dense(n // div_ * n // div_ * div_ * filters, activation="relu")(latent_inputs)
    x2 = layers.Reshape((n // div_, n // div_, div_ * filters))(x2)

    d2_filters = d2_filters * 2 ** layers_n
    for i in range(layers_n):
        x2 = layers.Conv2DTranspose(d2_filters, 3, activation="relu", strides=2, padding="same")(x2)
        d2_filters //= 2

    # d2_outputs = layers.Conv2DTranspose(c, 3, activation="sigmoid", strides=2, padding="same")(x2)
    d2_outputs = layers.Conv2DTranspose(c, 3, activation="sigmoid", padding="same")(x2)

    decoder = keras.Model(latent_inputs, [d1_outputs, d2_outputs], name="decoder")

    return decoder


def build_dual_vae(input_shape=(32, 32, 1), inter_dim=16, latent_dim=32, filters=16, layers_n=3, merge_type='add'):
    encoder = build_dual_encoder(input_shape=input_shape, inter_dim=inter_dim, latent_dim=latent_dim,
                                 filters=filters, layers_n=layers_n, merge_type=merge_type)
    decoder = build_dual_decoder(input_shape=input_shape, latent_dim=latent_dim, filters=filters, layers_n=layers_n)
    vae = VAE(encoder, decoder)
    return vae


if __name__ == '__main__':
    pass
