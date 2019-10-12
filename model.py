# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:11:36 2019

@author: 213539359
"""
from typing import Tuple

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, \
    Dropout, BatchNormalization, Conv2DTranspose
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from numpy.core._multiarray_umath import ndarray
from scipy import misc, ndimage
from PIL import Image
from skimage import data, img_as_float
from skimage import exposure
from keras.layers.core import Activation, Reshape

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

#from src 
import dataset_loader


"""================================CNN AUTOENCODER=========================="""


def create_model():
    # input_img = Input(shape=(720, 576, 1))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(120, 120, 1))

    # 120
    x = Conv2D(256, (3, 3), padding='same')(input_img)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # 60
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # 30 == 32
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)
    
    encoded = MaxPooling2D((2, 2), padding='same')(x)
    
    # 15 == 16
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # 128
    x = Conv2DTranspose(32, (3, 3), padding='same')(encoded)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    # 32
    x = Conv2DTranspose(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    # 64
    x = Conv2DTranspose(128, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation('relu')(x)

    x = UpSampling2D((2, 2))(x)

    # 32
    x = Conv2DTranspose(256, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation('relu')(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)
    
    autoencoder.summary()
    
    return autoencoder, encoder


"""================================TRAINING=========================="""


def training(autoencoder, encoder):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Adam(lr=0.01)
    autoencoder.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    x_train, x_test = dataset_loader.load_data_wrapper()
    train_set, test_set, valid_set = dataset_loader.data_preprocessing(x_train, x_test, augment=1000)

    early = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=50,
        mode='min',
        verbose=1
    )

    autoencoder.fit(train_set[0], train_set[1],
                    epochs=500,
                    batch_size=32,
                    shuffle=True,
                    validation_data=(test_set[0], test_set[1]),
                    callbacks=[early]
                    )
    # TensorBoard(log_dir='C:/Users/213539359/Downloads/AlexNet-Tensorflow-master/logs/'),
    #autoencoder.save('C:/Users/213539359/Downloads/AlexNet-Tensorflow-master/model/autoencoder_3.h5')

    encoded_img = encoder.predict(test_set[0])
    decoded_imgs = autoencoder.predict(test_set[0])

    results = autoencoder.evaluate(test_set[0], test_set[1], batch_size=1)
    print("TEST LOSS, TEST ACC: ", results)

    return x_test, decoded_imgs, encoded_img


"""====================================DISPLAY=============================="""


def display(x_test, decoded_imgs, encoded_imgs):
    shape = (120, 120)
    n = 20  # how many digits we will display
    plt.figure(figsize=(20, 4))
    for i in range(n):
        # display original
        ax = plt.subplot(2, n, i + 1)
        plt.imshow(x_test[0][i].reshape(shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        ax = plt.subplot(2, n, i + 1 + n)
        plt.imshow(decoded_imgs[i].reshape(shape))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # display reconstruction
        """
        ax = plt.subplot(2, n, ((2 * n) + i + 1))
        plt.imshow(encoded_imgs[i].reshape(4,4 ))
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)
        """
    plt.show()


if __name__ == '__main__':
    start_time = time.time()

    autoencoder, encoder = create_model()
    x_test, decoded_imgs, encoded_img = training(autoencoder, encoder)
    display(x_test, decoded_imgs, encoded_img)

    end_time = time.time()
    print("====================================\t\
          PROCESSING TIME %s\t===================================="
          % str(end_time - start_time))
