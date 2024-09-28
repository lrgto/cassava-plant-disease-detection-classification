from ctypes import sizeof
import os
import shutil
import pandas
import tensorflow as tf
import zipfile
import numpy as np 

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array
from keras.layers import Rescaling
#careful, this deletes Cassava folder
archive = 'Cassava_plant_disease_-_Uniform.zip'
dirTo = 'Cassava/'
 
def fresh_setup():
    remove()
    extract()

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

def loadData(train):
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
    #train
    dataset = tf.keras.preprocessing.image_dataset_from_directory(
        'Cassava/utrain_train',
        labels="inferred",
        label_mode="int",
        color_mode="rgb",
        batch_size=1,
        image_size=(256, 256),
    )

    return dataset

    if(train):
        print('loading training data')
        get_data(train_data, CBB_train.loc[:,"image_id"], train_labels, 0,train_path)
        print(20)
        get_data(train_data, CBSD_train.loc[:,"image_id"], train_labels, 1,train_path)
        print(40)
        get_data(train_data, CGM_train.loc[:,"image_id"], train_labels, 2,train_path)
        print(60)
        get_data(train_data, CMD_train.loc[:,"image_id"], train_labels, 3,train_path)
        print(80)
        get_data(train_data, HEALTHY_train.loc[:,"image_id"], train_labels, 4,train_path)
        print(100)
        return np.array(train_data), np.array(train_labels)
    #test
    else:
        print('loading testing data')
        get_data(test_data, CBB_test.loc[:,"image_id"], test_labels, 0,dest_path)
        print(20)
        get_data(test_data, CBSD_test.loc[:,"image_id"], test_labels, 1,dest_path)
        print(40)
        get_data(test_data, CGM_test.loc[:,"image_id"], test_labels, 2,dest_path)
        print(60)
        get_data(test_data, CMD_test.loc[:,"image_id"], test_labels, 3,dest_path)
        print(80)
        get_data(test_data, HEALTHY_test.loc[:,"image_id"], test_labels, 4,dest_path)
        print(100)
        return np.array(test_data), np.array(test_labels)


def readImage(file):
    #img = tf.image.decode_jpeg(tf.io.read_file(file))
    #return tf.image.resize_with_crop_or_pad(img, 300,300)
    img = img_to_array(load_img(file))
    return img

def get_data(data, files, labels, label, directory):

    for file in files:
        #need to change to actual image data
        data.append(readImage(directory+file))
        labels.append(label)


