#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Apr 26 11:16:43 2019

@author: kantareddy
"""
import os
import tensorflow as tf
import cv2
import glob
from sklearn.utils import shuffle
import numpy as np
from sklearn.model_selection import train_test_split
import numpy
from keras.models import Sequential
from keras.layers import Dense
from keras.wrappers.scikit_learn import KerasRegressor
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from keras.models import Sequential
from keras.layers import Dense, Conv2D, Dropout, Flatten, MaxPooling2D


train_path='/home/kantareddy/Desktop/rotation/Regression_Problem/training_data/'
image_size=128
classes=os.listdir('/home/kantareddy/Desktop/rotation/Regression_Problem/training_data')
def load_train(train_path, image_size, classes):
    images = []
    labels = []
    img_names = []

    print('Going to read training images')
    for fields in classes:   
        index = classes.index(fields)
        print('Now going to read {} files (Index: {})'.format(fields, index))
        path = os.path.join(train_path, fields, '*g')
        files = glob.glob(path)
        for fl in files:
            image = cv2.imread(fl)
            image = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)
            image = image.astype(np.float32)
            #image = np.multiply(image, 1.0 / 255.0)
            images.append(image)
# =============================================================================
#             label = np.zeros(len(classes))
#             label[index] = 1.0
# =============================================================================
            labels.append(np.float32(fields))
            flbase = os.path.basename(fl)
            img_names.append(flbase)
    images = np.array(images)
    labels = np.array(labels)
# =============================================================================
#     labels=np.reshape(labels,(-1,1))
# =============================================================================
    img_names = np.array(img_names)

    return images, labels, img_names

images, labels, img_names = load_train(train_path, image_size, classes)
images, labels, img_names = shuffle(images, labels, img_names) 
x_train,x_test,y_train,y_test=train_test_split(images,labels, test_size=0.33)
x_train = x_train.reshape(x_train.shape[0], 128, 128, 3)
x_test = x_test.reshape(x_test.shape[0], 128, 128, 3) 

from keras import backend as K

def r2_keras(y_true, y_pred):
    SS_res =  K.sum(K.square(y_true - y_pred)) 
    SS_tot = K.sum(K.square(y_true - K.mean(y_true))) 
    return ( 1 - SS_res/(SS_tot + K.epsilon()) )


#x_train, x_test = x_train / 255.0, x_test / 255.0
# =============================================================================
# model = tf.keras.models.Sequential([
#     tf.keras.layers.Flatten(input_shape=(128, 128)),
#     tf.keras.layers.Dense(512, activation=tf.nn.relu),
#     tf.keras.layers.Dropout(0.2),
#     tf.keras.layers.Dense(1, activation=tf.keras.activations.linear)
#   ])
# =============================================================================
    
input_shape=(128, 128,3)
model = Sequential()
model.add(Conv2D(32, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Conv2D(64, kernel_size=(5,5), input_shape=input_shape))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Flatten()) # Flattening the 2D arrays for fully connected layers
model.add(Dense(1024, activation=tf.nn.relu))
model.add(Dropout(0.2))
model.add(Dense(1, activation=tf.keras.activations.linear))
model.compile(optimizer='adam',
                loss='mean_squared_error',
                metrics=[r2_keras])


# =============================================================================
# x_train, x_test = x_train / 255.0, x_test / 255.0
# model = tf.keras.models.Sequential([
#      tf.keras.layers.Flatten(input_shape=(128, 128)),
#      tf.keras.layers.Dense(512, activation=tf.nn.relu),
#      tf.keras.layers.Dropout(0.2),
#      tf.keras.layers.Dense(4, activation=tf.nn.softmax)
#    ])
# model.compile(optimizer='adam',
#                  loss='categorical_crossentropy',
#                  metrics=['accuracy'])
# =============================================================================
  
model.fit(x_train, y_train, epochs=550)
model.evaluate(x_test, y_test)
y_pred=model.predict(y_test)
images=[]
image = cv2.imread('/home/kantareddy/Desktop/rotation/Regression_Problem/testing_data/53/frame1.jpg')
image1 = cv2.resize(image, (image_size, image_size),0,0, cv2.INTER_LINEAR)

images.append(image1)
images = np.array(images, dtype=np.uint8)
images = images.astype('float32')
#images = np.multiply(images, 1.0/255.0) 
model.predict(images)
  
