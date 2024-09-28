import tensorflow as tf
from tensorflow import keras
from keras import layers, models
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np
import pickle
import setup
model = pickle.load('model.sav')
test_dataset = setup.loadDataset('test')
test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
print(test_acc)