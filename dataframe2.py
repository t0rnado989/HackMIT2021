import os
from PIL import Image, ImageChops
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split

tumors = "./brain_tumor_dataset/yes"
healthy = "./brain_tumor_dataset/no"
directories = [tumors, healthy]
classes = ["Brain Tumor", "Healthy"]
filepaths = []
labels = []
for dir, cl in zip(directories, classes):
    file_list = os.listdir(dir)
    for file in file_list:
        fpath = os.path.join(dir, file)
        filepaths.append(fpath)
        labels.append(cl)
file_paths = pd.Series(filepaths, name="file_paths")
label_series = pd.Series(labels, name="labels")
df = pd.concat([file_paths, label_series], axis=1)
df = pd.DataFrame(np.array(df).reshape(253, 2), columns = ['file_paths', 'labels'])

train_df, test_df = train_test_split(df, train_size=0.8, random_state=0)
train_df, valid_df = train_test_split(df, train_size=0.7, random_state=0)

target_size = (300, 300)
batch_size = 64

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input, zoom_range=0.1, horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
train_gen = train_datagen.flow_from_dataframe(train_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')
valid_gen = test_datagen.flow_from_dataframe(valid_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')
test_gen = test_datagen.flow_from_dataframe(test_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')

base_model = tf.keras.applications.InceptionResNetV2(include_top=False, input_shape=(300, 300, 3))

model = tf.keras.Sequential([
    base_model,
    tf.keras.layers.GlobalAveragePooling2D(),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.BatchNormalization(),
    tf.keras.layers.Dropout(0.2),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

learning_rate = 0.0001
model.compile(loss='binary_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=learning_rate), metrics=['accuracy'])
EPOCHS = 25
history = model.fit(train_gen, validation_data=valid_gen, epochs=EPOCHS)

# summarize history for accuracy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

model.evaluate(test_gen)