"""
Adapted from keras example cifar10_cnn.py
"""
from __future__ import print_function
#from keras.datasets import cifar10
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ReduceLROnPlateau, EarlyStopping,CSVLogger
import keras
from keras import regularizers
import numpy as np
import resnet
import os
from PIL import Image
import matplotlib.pyplot as plt
import pylab as pl

batch_size = 240
nb_epoch = 40
nb_classes = 2
batch_size_val = 120 
nb_epoch_val = 40
learnRate=1.e-2
lr_decay=0.0
data_augmentation_switch = [False, True]
#data_augmentation = False
'''
reduceFactor = np.sqrt(0.1)
patience_plaute=10
patience_stop=10
learnRate=1.0
min_learningrate = 1.e-8
min_delta=1.e-4

lr_reducer = ReduceLROnPlateau(monitor='val_loss', factor=reduceFactor, cooldown=0, \
                               patience=patience_plaute,
                               mode='auto',min_lr=min_learningrate
                              )
early_stopper = EarlyStopping(min_delta=min_delta, patience=patience_stop)
csv_logger = CSVLogger('model.csv')
#'''

# input image dimensions
img_rows, img_cols = 32, 32
# The CIFAR10 images are RGB
img_channels = 3

#===================Method Define Start===============
def plotTrainResult(obj_result, nb_epoch):
    #obj_result = FirstTrain
    #nb_epoch = nb_epoch
    epoch = list(range(nb_epoch))
    objNameFig = plt.figure()
    plt.plot(epoch,obj_result.history['acc'], 'bo-.', label="Training accuracy")
    plt.plot(epoch,obj_result.history['val_acc'],'ro-.', label="validation accuracy")
    plt.legend(loc='lower right')
    final_accuracy = obj_result.history['acc'][-1]
    plt.annotate(r'acc='+str(final_accuracy), xy=(nb_epoch-1,final_accuracy), xytext=(nb_epoch-1+0.1,final_accuracy-0.2),fontsize = 16,arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xticks(np.arange(len(epoch)), epoch, rotation=45)
    pl.xlabel('epoch')
    pl.ylabel('acc')
    pl.axis([0.0,nb_epoch,0.0,1.0])
    plt.grid(True)
    os.system("rm -rf feedback_TrainValidation.png")
    plt.ion()
    objNameFig.savefig("feedback_TrainValidation.png")

def plotAugmentationResult(obj_result, obj_result_augmentation, nb_epoch):
    #obj_result = FirstTrain
    #obj_result_augmentation = Augmentation
    #nb_epoch = nb_epoch
    epoch = list(range(nb_epoch))
    objNameFig = plt.figure()
    plt.subplot(2,1,1)
    plt.plot(epoch,obj_result_augmentation.history['loss'], 'bo-.', label="Training loss")
    plt.plot(epoch,obj_result_augmentation.history['val_loss'],'ro-.', label="validation loss")
    plt.legend(loc='upper right')

    final_loss = obj_result_augmentation.history['loss'][-1]
    plt.annotate(r'loss='+str(final_loss), xy=(nb_epoch-1,final_loss), xytext=(nb_epoch-1+0.1,final_loss+0.2),fontsize = 16,arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xticks(np.arange(len(epoch)), epoch, rotation=45)
    pl.title('Using Data Augmentation')
    pl.xlabel('epoch')
    pl.ylabel('loss')
    pl.axis([0.0,nb_epoch,0.0,1.0])
    plt.grid(True)

    plt.subplot(2,1,2)
    plt.plot(epoch,obj_result.history['acc'], 'bo-.', label="Training accuracy")
    plt.plot(epoch,obj_result.history['val_acc'],'ro-.', label="validation accuracy")
    plt.legend(loc='lower right')

    final_accuracy = obj_result.history['acc'][-1]
    plt.annotate(r'acc='+str(final_accuracy), xy=(nb_epoch-1,final_accuracy), xytext=(nb_epoch-1+0.1,final_accuracy-0.2),fontsize = 16,arrowprops=dict(facecolor='black', shrink=0.05))
    plt.xticks(np.arange(len(epoch)), epoch, rotation=45)
    pl.xlabel('epoch')
    pl.ylabel('acc')
    pl.axis([0.0,nb_epoch,0.0,1.0])
    plt.grid(True)
    os.system("rm -rf feedback_DataAugmentation.png")
    plt.ion()
    objNameFig.savefig("feedback_DataAugmentation.png")

#===================Method Define End================

def get_files(file_dir):
    train = []
    label = []
    subDirList = [dirname for dirname in os.listdir(file_dir) \
                  if os.path.isdir(os.path.join(file_dir,dirname))]
    for subDir in subDirList:
        for filename in os.listdir(os.path.join(file_dir, subDir)):
            name = filename.split('.')
            image = Image.open(os.path.join(file_dir, subDir, filename))
            image = image.resize([img_rows, img_cols])
            image = np.array(image)
            train.append(image/255)
            if name[0]=='0':
                label.append([1,0])
            else:
                label.append([0,1])
    print('There are %d pic'%(len(label)))
    train = np.array(train)
    label = np.array(label)
    train = train.reshape(train.shape[0], img_rows, img_cols, img_channels)
    return train,label
# The data: train_set  validation_set
train_dir = "./picTrain"
validation_dir = "./picValidation"
train, train_label = get_files(train_dir)
validation, validation_label = get_files(validation_dir)

train = train.astype('float32')
validation = validation.astype('float32')

# subtract mean and normalize
mean_image = np.mean(train, axis=0)
train -= mean_image
validation -= mean_image

model = resnet.ResnetBuilder.build_resnet_50((img_channels, img_rows, img_cols), nb_classes)

keras.optimizers.Adam(
    lr=learnRate,
    beta_1=0.9, beta_2=0.999,
    epsilon=1.e-8,
    decay=lr_decay,
    amsgrad=False
)

model.compile(#loss='binary_crossentropy',
              loss='categorical_crossentropy',
              optimizer='Adam',
              #optimizer='Adadelta',
              metrics=['accuracy']
             )

#keras.optimizers.Adadelta(lr=learnRate, rho=0.95, epsilon=None, decay=0.0)

# This will do preprocessing and realtime data augmentation:
for data_augmentation in data_augmentation_switch:
    if not data_augmentation:
        print('Not using data augmentation.')
        train_datagen = ImageDataGenerator(
            rescale=1./255,
            shear_range=0.0,
            zoom_range=0.0,
            horizontal_flip=True
        )
        test_datagen = ImageDataGenerator(rescale=1./255)

        train_generator = train_datagen.flow_from_directory(
            directory=r"./picTrain/",
            target_size=(32, 32),
            batch_size=batch_size,
            color_mode="rgb",
            shuffle=True,
            seed=42,
            class_mode='categorical')

        validation_generator = test_datagen.flow_from_directory(
            directory=r"./picValidation/",
            target_size=(32, 32),
            batch_size=batch_size_val,
            color_mode="rgb",
            shuffle=True,
            seed=19,
            class_mode='categorical'
        )

        FirstTrain = model.fit_generator(
            generator=train_generator,
            steps_per_epoch=train.shape[0] // batch_size,
            epochs=nb_epoch,
            validation_data=validation_generator,
            validation_steps=nb_epoch
            #callbacks=[lr_reducer, early_stopper, csv_logger]
        )
        model.save('./parameter_save/weight.h5')
        plotTrainResult(FirstTrain, nb_epoch)
    else:
        print('Using real-time data augmentation.')
        datagen = ImageDataGenerator(
            featurewise_center=False,  # set input mean to 0 over the dataset
            samplewise_center=False,  # set each sample mean to 0
            featurewise_std_normalization=False,  # divide inputs by std of the dataset
            samplewise_std_normalization=False,  # divide each input by its std
            zca_whitening=False,  # apply ZCA whitening
            rotation_range=0,  # randomly rotate images in the range (degrees, 0 to 180)
            width_shift_range=0.1,  # randomly shift images horizontally (fraction of total width)
            height_shift_range=0.1,  # randomly shift images vertically (fraction of total height)
            horizontal_flip=True,  # randomly flip images
            vertical_flip=False)  # randomly flip images

        # Compute quantities required for featurewise normalization
        # (std, mean, and principal components if ZCA whitening is applied).
        datagen.fit(train)

        model.load_weights('./parameter_save/weight.h5', by_name=True)
        #'''
        # Fit the model on the batches generated by datagen.flow().
        Augmentation = model.fit_generator(datagen.flow(train, train_label, batch_size=batch_size),
                            steps_per_epoch=train.shape[0] // batch_size,
                            validation_data=(validation, validation_label),
                            epochs=nb_epoch, verbose=1, max_q_size=10
                            #callbacks=[lr_reducer, early_stopper, csv_logger]
                                     )
        model.save('./parameter_save/weight_augmentation.h5')
        plotAugmentationResult(FirstTrain, Augmentation, nb_epoch)
