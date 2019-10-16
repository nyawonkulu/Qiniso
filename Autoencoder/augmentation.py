# Save augmented images to file

import gzip
import pickle
import matplotlib.pyplot as plt
import numpy as np
from keras.datasets import mnist
from keras.preprocessing.image import ImageDataGenerator
from matplotlib import pyplot
import os

from skimage import exposure

import transformations


"""===================================DATA AUGMENTATION=============================="""
def aug():
    # load data
    print('\nAUGMENTING DATASET....')
    train, test = dataset_loader.load_data_wrapper()

    X_train = train[0]
    Y_train = train[1]

    X_train = np.reshape(X_train, (X_train.shape[0], 120, 120, 1)).astype('float32')
    Y_train = np.reshape(Y_train, (Y_train.shape[0], 120, 120, 1)).astype('float32')

    X_transformed, Y_transformed = transformations.augment(X_train, Y_train, n=20, elastic_alpha_range=20,
                                                           rotation_range=30,
                                                           shift_x_range=0.2, shift_y_range=0.2)

    datagen = ImageDataGenerator()

    empty = np.zeros((120, 120, 1))
    # fit parameters from data
    datagen.fit(X_transformed, augment=True)
    datagen.fit(Y_transformed, augment=True)
    # configure batch size and retrieve one batch of images

    main = []
    label = []
    
    batch = X_transformed.shape[0]

    for X_batch, y_batch in datagen.flow(X_transformed, Y_transformed, batch_size=batch, shuffle=False):
        for i in range(0, batch):
            if not np.array_equal(X_batch[i], empty):
                main.append(X_batch[i])                                              
                label.append(y_batch[i])
        break

    main = np.array(main)
    label = np.array(label)

    dump = [main, label]
    print('AUGMENTATION DONE....\n')
    return dump


"""===================================DUMP AUGMENTATION TO FILE=============================="""
def dump_file(trainset):
    f = gzip.open('augumented.pkl.gz', 'wb')
    print('\nDUMPING....')
    pickle.dump(trainset, f, protocol=2)
    f.close()
    print('DONE DUMPING....\n')


"""===================================LOAD AUGMENTED DATA=============================="""
def load_file():
    print('\nLOADING DATA....')
    f = gzip.open('augumented.pkl.gz', 'rb')
    training_data = pickle.load(f, encoding='latin1')
    f.close()
    print('DATA LOADED....\n')
    return training_data


"""===================================DATA PREPROCESSING=============================="""
def display_(training_data):
    plt.figure(figsize=(10, 10))
 
    for image in range(10):
        plt.imshow(training_data[0][image].reshape(120, 120))
        plt.show()
        
        plt.imshow(training_data[1][image].reshape(120, 120))
        plt.show()

