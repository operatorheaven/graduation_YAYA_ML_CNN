from __future__ import division, print_function, absolute_import  

import numpy as np
from PIL import Image 
import os
import time
import shutil
import matplotlib.pyplot as plt
import itertools

import keras_resnet

img_rows, img_cols = 128, 128

def buildModel():
    nb_classes = 2

    img_channels = 3
    model = keras_resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)
    model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])
    #establish  old model 
    model.load_weights('/home/boper/CNN/ResNet/keras_resnet/module/50_200_load_50/weight')
    return model

def predicMsk(picPath,model):
    # Data loading  
    test = []
    image = Image.open(picPath)
    image = image.resize([img_rows, img_cols])
    image = np.array(image)
    test.append(image/255)
    test = np.array(test)
    #print ('len.....len',len(test[0][0]))

    a = model.predict(test)
    return test
    #return np.argmax(a)


time_start=time.time()
rootdir= '/home/boper/CNN/pic/trainTest'
#rootdir= '/home/boper/CNN/pic/picTest/'
dst = '/home/boper/CNN/pic/newPic/'
pic = os.listdir(rootdir)

label = []
zero_zero = 0
zero_one = 0
one_zero = 0
one_one = 0

model = buildModel()