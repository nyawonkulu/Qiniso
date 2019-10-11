# -*- coding: utf-8 -*-
"""
Created on Fri Sep 13 02:11:36 2019

@author: 213539359
"""

from keras.layers import Input, Dense, Conv2D, MaxPooling2D, UpSampling2D, Cropping2D, ZeroPadding2D, \
    Dropout, BatchNormalization
from keras.models import Model
from keras.datasets import mnist
from keras import backend as K
from keras.callbacks import TensorBoard, EarlyStopping
from keras.optimizers import SGD, Adam
from keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
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


# from src
import leukemia_loader


DEFAULT_SIZE = (120, 120)

"""================================CNN AUTOENCODER=========================="""


def cnn_autoencoder():
    # input_img = Input(shape=(720, 576, 1))  # adapt this if using `channels_first` image data format
    input_img = Input(shape=(120, 120, 1))

    # 32
    x = Conv2D(32, (3, 3), padding='same')(input_img)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # 64
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = MaxPooling2D((2, 2), padding='same')(x)

    # 128
    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    encoded = MaxPooling2D((2, 2), padding='same')(x)

    # at this point the representation is (4, 4, 8) i.e. 128-dimensional

    # 128
    x = Conv2D(128, (3, 3), padding='same')(encoded)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = Conv2D(128, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)
    x = UpSampling2D((2, 2))(x)

    # 64
    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = Conv2D(64, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Dropout(0.2)(x)
    x = Activation("relu")(x)

    x = UpSampling2D((2, 2))(x)

    # 32
    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = Conv2D(32, (3, 3), padding='same')(x)
    x = BatchNormalization(momentum=0.1)(x)
    x = Activation("relu")(x)
    x = Dropout(0.2)(x)

    x = UpSampling2D((2, 2))(x)

    decoded = Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

    autoencoder = Model(input_img, decoded)

    encoder = Model(input_img, encoded)

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

        x_train_in = np.reshape(x_train[0], (len(x_train[0]), height, width, 1)).astype('float32') / 255
        x_train_out = np.reshape(x_train[1], (len(x_train[1]), height, width, 1)).astype('float32') / 255

        x_test_in = np.reshape(x_test[0], (len(x_test[0]), height, width, 1)).astype('float32') / 255
        x_test_out = np.reshape(x_test[1], (len(x_test[1]), height, width, 1)).astype('float32') / 255

        # x_train_in, x_train_out = data_augmentation(x_train_in, x_train_out, augment_size=1000)

        valid = int(len(x_train_in) * 0.01)

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
                    batch_size=8,
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
