import os
from PIL import Image, ImageChops
import pandas as pd
import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import sklearn.metrics as metrics

WIDTH = 200
HEIGHT = 200

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

target_size = (WIDTH, HEIGHT)
batch_size = 1

train_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input, zoom_range=0.1, horizontal_flip=True, width_shift_range=0.05, height_shift_range=0.05)
test_datagen = ImageDataGenerator(preprocessing_function=tf.keras.applications.inception_resnet_v2.preprocess_input)
train_gen = train_datagen.flow_from_dataframe(df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary', shuffle=False)
# valid_gen = test_datagen.flow_from_dataframe(valid_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')
# test_gen = test_datagen.flow_from_dataframe(test_df, x_col='file_paths', y_col='labels', target_size=target_size, batch_size=batch_size, color_mode='rgb', class_mode='binary')

loaded_model = tf.keras.models.load_model("Model_1.0", custom_objects=None, compile=True)

truths = []
count = 0
for image in df['labels']:
    if(image == "Healthy"):
        truths.append(1)
    elif (image == "Brain Tumor"):
        truths.append(0)
    # print(train_gen[count])
    # print(truths[count])
    count += 1

print(train_gen)

# calculate the fpr and tpr for all thresholds of the classification
train_gen.reset()
# probs = loaded_model.predict_generator(train_gen)
probs = loaded_model.predict(train_gen)
# print(probs, len(probs))
# print(len(train_gen))
fpr, tpr, threshold = metrics.roc_curve(truths, probs)
roc_auc = metrics.auc(fpr, tpr)

plt.title('Receiver Operating Characteristic')
plt.plot(fpr, tpr, 'b', linewidth = 2, label = 'AUC = %0.5f' % roc_auc)
plt.legend(loc = 'lower right')
plt.plot([0, 1], [0, 1],'r--')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.ylabel('True Positive Rate')
plt.xlabel('False Positive Rate')
plt.show()