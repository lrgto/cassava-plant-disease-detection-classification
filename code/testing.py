import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
from ctypes import sizeof
import os
import shutil
import pandas
import zipfile
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Rescaling

archive = 'Cassava_plant_disease_-_Uniform.zip'
dirTo = 'Cassava/'

def fresh_setup():
    remove()
    extract()
    prepareData()

def remove():
    try:
        shutil.rmtree('Cassava/')
    except:
        return

def extract():
    with zipfile.ZipFile(archive, 'r') as zip_ref:
        zip_ref.extractall(dirTo)
    print('done')

def move(files, dest):
    srcpath = 'Cassava/utrain_images/'
    
    for file in files:
        if(not os.path.exists(dest+file)):
            os.makedirs(dest+file)
        try:    #print(file)
            shutil.move(srcpath+file, dest+file)
        except FileNotFoundError:
            if os.path.exists(dest+file):
                #file already moved, can be ignored
                #print(file+' already moved')
                continue
            else:
                print('could not find '+file)
                continue


def loadDataset(type):
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Cassava/utrain_'+type,
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=1,
        image_size=(256, 256),
    )

    return dataset

def prepareData():
    labels = pandas.read_csv(dirTo+'utrain.csv')
    #Classes are :
    #0: 'Cassava Bacterial Blight (CBB)',
    #1: 'Cassava Brown Streak Disease (CBSD)',
    #2: 'Cassava Green Mottle (CGM)',
    #3: 'Cassava Mosaic Disease (CMD)',
    #4: 'Healthy'
    CBB = labels.loc[labels['labels'] == 0]
    CBSD = labels.loc[labels['labels'] == 1]
    CGM = labels.loc[labels['labels'] == 2]
    CMD = labels.loc[labels['labels'] == 3]
    HEALTHY = labels.loc[labels['labels'] == 4]

    CBB_train = CBB.iloc[:round(len(CBB)/2),:]
    CBB_test = CBB.iloc[round(len(CBB)/2):,:]
    CBSD_train = CBSD.iloc[:round(len(CBSD)/2),:]
    CBSD_test = CBSD.iloc[round(len(CBSD)/2):,:]
    CGM_train = CGM.iloc[:round(len(CGM)/2),:]
    CGM_test = CGM.iloc[round(len(CGM)/2):,:]
    CMD_train = CMD.iloc[:round(len(CMD)/2),:]
    CMD_test = CMD.iloc[round(len(CMD)/2):,:]
    HEALTHY_train = HEALTHY.iloc[:round(len(HEALTHY)/2),:]
    HEALTHY_test = HEALTHY.iloc[round(len(HEALTHY)/2):,:]
    move(CBB_test.loc[:,"image_id"], 'Cassava/utrain_test/CBB/')
    move(CBSD_test.loc[:,"image_id"], 'Cassava/utrain_test/CBSD/')
    move(CGM_test.loc[:,"image_id"], 'Cassava/utrain_test/CGM/')
    move(CMD_test.loc[:,"image_id"], 'Cassava/utrain_test/CMD/')
    move(HEALTHY_test.loc[:,"image_id"], 'Cassava/utrain_test/HEALTHY/')
    move(CBB_train.loc[:,"image_id"], 'Cassava/utrain_train/CBB/')
    move(CBSD_train.loc[:,"image_id"], 'Cassava/utrain_train/CBSD/')
    move(CGM_train.loc[:,"image_id"], 'Cassava/utrain_train/CGM/')
    move(CMD_train.loc[:,"image_id"], 'Cassava/utrain_train/CMD/')
    move(HEALTHY_train.loc[:,"image_id"], 'Cassava/utrain_train/HEALTHY/')
    train_data = []
    train_labels = []
    test_data = []
    test_labels = []
    train_path = 'Cassava/utrain_images/'
    dest_path = 'Cassava/utrain_test/'


physical_devices = tf.config.experimental.list_physical_devices('GPU')
tf.config.experimental.set_memory_growth(physical_devices[0], True)
#tf.config.set_visible_devices([], 'GPU')
fresh_setup()
dataset = loadDataset("train")

#model 1
def model1():
    model = keras.Sequential()
    model.add(Conv2D(filters=32,
                activation='relu', 
                kernel_size=(2,2), 
                strides=(1,1),
                padding='same',
                input_shape=(256,256,3),
                data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(filters=64,
                activation='relu',
                kernel_size=(2,2),
                strides=(1,1),
                padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Flatten())        
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    return model

def model2():
    model = keras.Sequential()
    model.add(Conv2D(filters=32,
                activation='relu', 
                kernel_size=(2,2), 
                strides=(1,1),
                padding='same',
                input_shape=(256,256,3),
                data_format='channels_last'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Conv2D(filters=64,
                activation='relu',
                kernel_size=(2,2),
                strides=(1,1),
                padding='valid'))
    model.add(MaxPooling2D(pool_size=(2,2),strides=2))
    model.add(Flatten())        
    model.add(Dense(64, activation='relu'))
    model.add(Dense(5, activation='softmax'))
    return model

models = [None]*2
models[0] = model1()
models[1] = model2()

import time
#testing models

def printToFile(label, epochs, timeTaken, accuracy, loss):
    f = open("results.txt", "a")
    f.write('Model ' + label + "Epochs: "+epochs)
    f.write("Completed in " + timeTaken + " seconds")
    f.write("Testing accuracy: " + accuracy)
    f.write("Testing loss: " + loss)
    f.write("")
    f.close()

for i in range(0,len(models)):
    start = int(time.time())
    model = models[i]
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(dataset,batch_size=32, verbose=2, epochs=1)
    model.save("models/model/1/"+i+1)
    test_dataset = loadDataset('test')
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    print("test accuracy")
    print(test_acc)
    finish = int(time.time())
    printToFile(i, 1, finish-start, test_acc, test_loss)

for i in range(0,len(models)):
    start = int(time.time())
    model = models[i]
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(dataset,batch_size=32, verbose=2, epochs=5)
    model.save("models/model/5/"+i+1)
    test_dataset = loadDataset('test')
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    print("test accuracy")
    print(test_acc)
    finish = int(time.time())
    printToFile(i, 5, finish-start, test_acc, test_loss)

for i in range(0,len(models)):
    start = int(time.time())
    model = models[i]
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    print(model.summary())
    history = model.fit(dataset,batch_size=32, verbose=2, epochs=15)
    model.save("models/model/15/"+i+1)
    test_dataset = loadDataset('test')
    test_loss, test_acc = model.evaluate(test_dataset, verbose=2)
    print("test accuracy")
    print(test_acc)
    finish = int(time.time())
    printToFile(i, 15, finish-start, test_acc, test_loss)


