import os
from PIL import Image
import pandas as pd
import numpy as np
import tensorflow as tf

directory = "./brain_tumor_dataset/no"
WIDTH = 500
HEIGHT = 500
SAMPLES = 253
VALIDATION_PERCENTAGE = 0.2

dataframe_n = pd.DataFrame()
for file in os.listdir(directory):
    try:
        with Image.open(directory + "/" + file) as image:
            image = image.convert("L")
            image = image.resize((WIDTH, HEIGHT))
            pixels = list(image.getdata())
            pixels.append(0)
            array = np.array(pixels)
            df = pd.DataFrame(array).transpose()
            dataframe_n = dataframe_n.append(df)
    except OSError:
        pass

directory = "./brain_tumor_dataset/yes"

dataframe_y = pd.DataFrame()
for file in os.listdir(directory):
    try:
        with Image.open(directory + "/" + file) as image:
            image = image.convert("L")
            image = image.resize((WIDTH, HEIGHT))
            pixels = list(image.getdata())
            pixels.append(1)
            array = np.array(pixels)
            df = pd.DataFrame(array).transpose()
            dataframe_y = dataframe_y.append(df)
    except OSError:
        pass

def create_model():
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(250000,)),
        tf.keras.layers.Dense(256, activation=tf.keras.activations.relu),
        tf.keras.layers.Dense(1, activation=tf.keras.activations.softmax)
    ])
    return model

dataframe = dataframe_n.append(dataframe_y, ignore_index=True)

target = dataframe[WIDTH*HEIGHT]
dataframe = dataframe.drop([WIDTH*HEIGHT], axis=1)

validation_samples = int(VALIDATION_PERCENTAGE * SAMPLES)

validation_indices = np.random.choice(dataframe.index, validation_samples, replace=False)
validation_dataset = dataframe.iloc[validation_indices]
validation_output = target.iloc[validation_indices]
training = dataframe.drop(validation_indices)
training_output = target.drop(validation_indices)

validation_dataset = validation_dataset.reset_index()
validation_output = validation_output.reset_index()
training = training.reset_index()
training_output = training_output.reset_index()

validation_dataset = validation_dataset.drop(['index'], axis=1)
validation_output = validation_output.drop(['index'], axis=1)
training = training.drop(['index'], axis=1)
training_output = training_output.drop(['index'], axis=1)

model = create_model()
EPOCHS = 20
model.compile(optimizer=tf.keras.optimizers.Adam(), loss=tf.keras.losses.binary_crossentropy, metrics=['accuracy'])
model.fit(x=training.values, y=training_output.values, epochs=EPOCHS,
          validation_data=(validation_dataset.values, validation_output.values),
          use_multiprocessing=True)
print(validation_output)
print(model(validation_output.values))











# for i in range(0, len(file_names), 5):
#     dataframe = pd.DataFrame()
#     for j in range(i * 5, (i * 5) + 5):
#         try:
#             with Image.open(file_names[i]) as image:
#                 print(image.format, f"{image.size}x{image.mode}")
#                 image = image.convert("L")
#                 image = image.resize((50, 50))
#                 pixels = list(image.getdata())
#                 print(len(pixels))
#                 array = np.array(pixels)
#                 # print(array)
#                 df = pd.DataFrame(array).transpose()
#                 print(df)
#                 dataframe = dataframe.append(df)
#         except OSError:
#             pass
#     print(dataframe)

