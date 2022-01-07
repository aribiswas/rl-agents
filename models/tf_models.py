"""
Deep Q network model (TensorFlow 2.x).
@author: abiswas
"""

import tensorflow as tf


def qnetwork(input_shape, num_outputs, is_image_obs=False):
    if is_image_obs:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Conv2D(32, 4, strides=(2, 2), activation='relu', padding='valid', input_shape=input_shape),
            tf.keras.layers.Conv2D(64, 2, strides=(1, 1), activation='relu', padding='valid', input_shape=input_shape),
            tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2)),
            tf.keras.layers.Dropout(rate=0.5),
            tf.keras.layers.Flatten(),
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_outputs)])
    else:
        model = tf.keras.models.Sequential([
            tf.keras.layers.Dense(64, activation=tf.nn.tanh, input_shape=input_shape),
            tf.keras.layers.Dense(64, activation=tf.nn.tanh),
            tf.keras.layers.Dense(num_outputs)])
    model.compile(optimizer=tf.optimizers.Adam(0.01), loss='mse')
    return model
