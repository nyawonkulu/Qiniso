# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:11:36 2019

@author: 213539359
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, Dropout
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from scipy import misc, ndimage
from PIL import Image
from skimage import data, img_as_float
from skimage import exposure

import numpy as np
import matplotlib.pyplot as plt
import time
import os
import argparse

from keras.models import Model
from keras.layers import Input
from keras.layers.core import Activation, Reshape
from keras.layers.convolutional import Conv2D
from keras.layers.normalization import BatchNormalization




# from src
import leukemia_loader

DEFAULT_SIZE = (120, 120)

"""================================CNN AUTOENCODER=========================="""


def cnn_autoencoder():
    # input_img = Input(shape=(720, 576, 1))  # adapt this if using `channels_first` image data format

    output_mode='softmax'
    inputs = Input(shape=(120, 120, 1))
     # 120

    conv_1 = Conv2D(64, (3, 3), padding='same')(inputs)
    conv_1 = BatchNormalization()(conv_1)
    conv_1 = Activation('relu')(conv_1)
    conv_2 = Conv2D(64, (3, 3), padding='same')(conv_1)
    conv_2 = BatchNormalization()(conv_2)
    conv_2 = Activation('relu')(conv_2)

    pool_1 = MaxPooling2D((2, 2))(conv_2)

    conv_3 = Conv2D(128, (3, 3), padding='same')(pool_1)
    conv_3 = BatchNormalization()(conv_3)
    conv_3 = Activation('relu')(conv_3)
    conv_4 = Conv2D(128, (3, 3), padding='same')(conv_3)
    conv_4 = BatchNormalization()(conv_4)
    conv_4 = Activation('relu')(conv_4)

    pool_2 = MaxPooling2D((2, 2))(conv_4)

    padding = ZeroPadding2D(padding=(1, 1), dim_ordering='default')(pool_2)
    conv_5 = Conv2D(256, (3, 3), padding='same')(padding)
    conv_5 = BatchNormalization()(conv_5)
    conv_5 = Activation('relu')(conv_5)
    conv_6 = Conv2D(256, (3, 3), padding='same')(conv_5)
    conv_6 = BatchNormalization()(conv_6)
    conv_6 = Activation('relu')(conv_6)
    conv_7 = Conv2D(256, (3, 3), padding='same')(conv_6)
    conv_7 = BatchNormalization()(conv_7)
    conv_7 = Activation('relu')(conv_7)

    pool_3 = MaxPooling2D((2, 2))(conv_7)

    conv_8 = Conv2D(512, (3, 3), padding='same')(pool_3)
    conv_8 = BatchNormalization()(conv_8)
    conv_8 = Activation('relu')(conv_8)
    conv_9 = Conv2D(512, (3, 3), padding='same')(conv_8)
    conv_9 = BatchNormalization()(conv_9)
    conv_9 = Activation('relu')(conv_9)
    conv_10 = Conv2D(512, (3, 3), padding='same')(conv_9)
    conv_10 = BatchNormalization()(conv_10)
    conv_10 = Activation('relu')(conv_10)

    pool_4 = MaxPooling2D((2, 2))(conv_10)

    conv_11 = Conv2D(512, (3, 3), padding='same')(pool_4)
    conv_11 = BatchNormalization()(conv_11)
    conv_11 = Activation('relu')(conv_11)
    conv_12 = Conv2D(512, (3, 3), padding='same')(conv_11)
    conv_12 = BatchNormalization()(conv_12)
    conv_12 = Activation('relu')(conv_12)
    conv_13 = Conv2D(512, (3, 3), padding='same')(conv_12)
    conv_13 = BatchNormalization()(conv_13)
    conv_13 = Activation('relu')(conv_13)

    pool_5 = MaxPooling2D((2, 2))(conv_13)
    print("Build enceder done..")

    # decoder

    unpool_1 =UpSampling2D((2, 2))(pool_5)

    conv_14 = Conv2D(512, (3, 3), padding='same')(unpool_1)
    conv_14 = BatchNormalization()(conv_14)
    conv_14 = Activation('relu')(conv_14)
    conv_15 = Conv2D(512, (3, 3), padding='same')(conv_14)
    conv_15 = BatchNormalization()(conv_15)
    conv_15 = Activation('relu')(conv_15)
    conv_16 = Conv2D(512, (3, 3), padding='same')(conv_15)
    conv_16 = BatchNormalization()(conv_16)
    conv_16 = Activation('relu')(conv_16)

    unpool_2 =UpSampling2D((2, 2))(conv_16)

    conv_17 = Conv2D(512, (3, 3), padding='same')(unpool_2)
    conv_17 = BatchNormalization()(conv_17)
    conv_17 = Activation('relu')(conv_17)
    conv_18 = Conv2D(512, (3, 3), padding='same')(conv_17)
    conv_18 = BatchNormalization()(conv_18)
    conv_18 = Activation('relu')(conv_18)
    conv_19 = Conv2D(256, (3, 3), padding='same')(conv_18)
    conv_19 = BatchNormalization()(conv_19)
    conv_19 = Activation('relu')(conv_19)

    unpool_3 =UpSampling2D((2, 2))(conv_19)

    padding = Cropping2D(cropping=((1, 1), (1, 1)))(unpool_3)
    conv_20 = Conv2D(256, (3, 3), padding='same')(padding)
    conv_20 = BatchNormalization()(conv_20)
    conv_20 = Activation('relu')(conv_20)
    conv_21 = Conv2D(256, (3, 3), padding='same')(conv_20)
    conv_21 = BatchNormalization()(conv_21)
    conv_21 = Activation('relu')(conv_21)
    conv_22 = Conv2D(128, (3, 3), padding='same')(conv_21)
    conv_22 = BatchNormalization()(conv_22)
    conv_22 = Activation('relu')(conv_22)

    unpool_4 =UpSampling2D((2, 2))(conv_22)

    conv_23 = Conv2D(128, (3, 3), padding='same')(unpool_4)
    conv_23 = BatchNormalization()(conv_23)
    conv_23 = Activation('relu')(conv_23)
    conv_24 = Conv2D(64, (3, 3), padding='same')(conv_23)
    conv_24 = BatchNormalization()(conv_24)
    conv_24 = Activation('relu')(conv_24)

    unpool_5 =UpSampling2D((2, 2))(conv_24)

    conv_25 = Conv2D(64, (3, 3), padding='same')(unpool_5)
    conv_25 = BatchNormalization()(conv_25)
    conv_25 = Activation('relu')(conv_25)

    conv_26 = Conv2D(1, (1, 1), padding="valid")(conv_25)
    conv_26 = BatchNormalization()(conv_26)

    outputs = Activation(output_mode)(conv_26)

    autoencoder = Model(inputs, outputs)

    encoder = Model(inputs, pool_5)

    training(autoencoder, encoder)


"""===================================DATA AUGMENTATION==============================="""


def data_augmentation(x_train_in, x_train_out, augment_size):
    def histogram_equalization(x):
        if np.random.random() < 0.5:
            x = exposure.equalize_hist(x)

    def adaptive_equalization(x):
        if np.random.random() < 0.5:
            x = exposure.equalize_adapthist(x, clip_limit=0.01)

    def contrast_stretching(x):
        if np.random.random() < 0.5:
            p2, p98 = np.percentile(x, (2, 98))
            x = exposure.rescale_intensity(x, in_range=(p2, p98))

    def to_lab(x):
        if np.random.random() < 0.5:
            x = exposure.rgb2lab(x, illuminant='D65', observer='2')


    image_generator = ImageDataGenerator(
        #rescale=1.0/255.0,
        rotation_range=10,
        #shear_range=0.8,
        featurewise_center=False,
        samplewise_std_normalization=False,
        zoom_range=0.05,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        #preprocessing_function=histogram_equalization
    )

    # fit data for zca whitening
    x_train_in, x_train_out
    image_generator.fit(x_train_in, augment=True)
    # get transformed images
    randidx = np.random.randint(x_train_in.shape[0], size=augment_size)
    x_augmented = x_train_in[randidx].copy()
    y_augmented = x_train_out[randidx].copy()
    x_augmented = image_generator.flow(x_augmented, np.zeros(augment_size),
                                batch_size=augment_size, shuffle=False).next()[0]
    # append augmented data to trainset
    x_train = np.concatenate((x_train_in, x_augmented))
    y_train = np.concatenate((x_train_out, y_augmented))

    return x_train, y_train


"""===================================DATA PREPROCESSING=============================="""


def data_preprocessing(x_train, x_test):
        height = 120
        width = 120

        x_train_in, x_train_out = data_augmentation(x_train[0], x_train[1], augment_size=10000)
        """
        x_train_in = np.reshape(x_train[0], (len(x_train[0]), height, width, 1)).astype('float32') / 255
        x_train_out = np.reshape(x_train[1], (len(x_train[1]), height, width, 1)).astype('float32') / 255
        """
        x_train_in = np.reshape(x_train_in, (len(x_train_in), height, width, 1)).astype('float32') / 255
        x_train_out = np.reshape(x_train_out , (len(x_train_out ), height, width, 1)).astype('float32') / 255

        x_test_in = np.reshape(x_test[0], (len(x_test[0]), height, width, 1)).astype('float32') / 255
        x_test_out = np.reshape(x_test[1], (len(x_test[1]), height, width, 1)).astype('float32') / 255



        valid = int(len(x_train_in) * 0.2)

        x_valid_in = x_train_in[-valid:]
        x_valid_out = x_train_out[-valid:]

        x_train_in = x_train_in[:-valid]
        x_train_out = x_train_out[:-valid]

        train_set = (x_train_in, x_train_out)
        test_set = (x_test_in, x_test_out)
        valid_set = (x_valid_in, x_valid_out)

        return train_set, test_set, valid_set


"""===================================TRAINING=============================="""


def training(autoencoder, encoder):
    sgd = SGD(lr=0.01, decay=1e-6, momentum=0.9, nesterov=True) #Adam(lr=0.01)
    autoencoder.compile(optimizer=sgd, loss='binary_crossentropy', metrics=['accuracy'])

    x_train, x_test = leukemia_loader.load_data_wrapper()
    train_set, test_set, valid_set = data_preprocessing(x_train, x_test)

    autoencoder.summary()

    early = EarlyStopping(
        monitor='val_loss',
        min_delta=0,
        patience=20,
        mode='min',
        verbose=1

    )
    autoencoder.fit(train_set[0], train_set[1],
                    epochs=500,
                    batch_size=256,
                    shuffle=True,
                    validation_data=(valid_set[0], valid_set[1]),
                    callbacks=[early]
                    )
    # TensorBoard(log_dir='C:/Users/213539359/Downloads/AlexNet-Tensorflow-master/logs/'),
    #autoencoder.save('C:/Users/213539359/Downloads/AlexNet-Tensorflow-master/model/autoencoder_3.h5')

    encoded_img = encoder.predict(test_set[0])
    decoded_imgs = autoencoder.predict(test_set[0])

    results = autoencoder.evaluate(test_set[0], test_set[1], batch_size=1)
    print("TEST LOSS, TEST ACC: ", results)

    display(x_test, decoded_imgs, encoded_img)


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


def resize_image(input_dir, size=DEFAULT_SIZE):
    try:
        img = Image.open(input_dir)
        img = img.resize((size[0], size[1]), Image.LANCZOS)
        plt.imshow(img)
        plt.show()
        exit(0)
    except IOError:
        print("unable to resize image")

    return

if __name__ == '__main__':
    start_time = time.time()
    # data_aug()
    # resize_image('C:/Users/213539359/Downloads/AlexNet-Tensorflow-master/src/001.bmp', DEFAULT_SIZE)
    cnn_autoencoder()
    end_time = time.time()
    print("====================================\t\
          PROCESSING TIME %s\t===================================="
          % str(end_time - start_time))
