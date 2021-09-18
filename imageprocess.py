import pandas as pd
import numpy as np
import tensorflow as tf
import os

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250000,)),
        tf.keras.layers.Dense(20000, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)
    ])
    return model


