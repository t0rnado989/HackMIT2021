import SimpleITK as sitk
import numpy as np
import tensorflow as tf
from tensorflow import keras

image_test = 'add_image-here.nii' # FIXME, read.nii images

def _int64_feature(value):
    return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))

def _float_feature(value):
    return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))

train_filename = 'tensorflow_records'

writer = tf.python_io.TFRecordWriter(train_filename)

# for meta_data in all_filenames:   #FIXME
#     img, label = load_img(meta_data, reader_params)


