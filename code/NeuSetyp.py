import os
import shutil
import tensorflow as tf
import pandas as pd
import numpy as np
from keras import layers
import numpy as np
import PIL
import PIL.Image
import pathlib

img_height = 800
img_width = 600

train_ds = tf.keras.utils.image_dataset_from_directory(
  'Cyssava',
  labels='inferred',
  label_mode='categorical',
  validation_split=0.2,
  subset="training",
  batch_size=16,
  seed=123,
  image_size=(img_height, img_width))

val_ds = tf.keras.utils.image_dataset_from_directory(
  'Cyssava',
  labels='inferred',
  label_mode='categorical',
  validation_split=0.2,
  subset="validation",
  batch_size=16,
  seed=123,
  image_size=(img_height, img_width))

num_classes = 5

model = tf.keras.Sequential([
  tf.keras.layers.Rescaling(1./255),
  tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(None, 800, 600, 3)),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Conv2D(32, 3, activation='relu'),
  tf.keras.layers.MaxPooling2D(),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(128, activation='relu'),
  tf.keras.layers.Dense(num_classes)
])

model.compile(
  optimizer='adam',
  loss=tf.losses.CategoricalCrossentropy(from_logits=True),
  metrics=['accuracy'])

model.fit(
  train_ds,
  validation_data=val_ds,
  batch_size=16,
  epochs=3
)
