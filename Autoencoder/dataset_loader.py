"""
mnist_loader
~~~~~~~~~~~~

A library to load the MNIST image data.  For details of the data
structures that are returned, see the doc strings for ``load_data``
and ``load_data_wrapper``.  In practice, ``load_data_wrapper`` is the
function usually called by our neural network code.
"""

#### Libraries
# Standard library
import pickle
import gzip

# Third-party libraries
import numpy as np
from keras_preprocessing.image import ImageDataGenerator
from skimage import exposure
import augmentation

"""===================================LOAD DATA==============================="""


def load_data():
    f = gzip.open('./file_out.pkl[90].gz', 'rb')
    training_data, test_data = pickle.load(f, encoding='latin1')
    f.close()
    return training_data, test_data


"""===================================LOAD DATA WRAPPER==============================="""


def load_data_wrapper():
    tr_d, te_d = load_data()

    shape = 14400, 1

    training_inputs = [np.reshape(x, shape) for x in tr_d[0]]
    training_out = [np.reshape(x, shape) for x in tr_d[1]]

    test_inputs = [np.reshape(x, shape) for x in te_d[0]]
    test_out = [np.reshape(x, shape) for x in te_d[1]]

    train_set = (np.array(training_inputs), np.array(training_out))
    test_set = (np.array(test_inputs), np.array(test_out))

    return train_set, test_set


"""===================================DATA AUGMENTATION==============================="""


def data_augmentation():
    dataset = augmentation.load_file()
    return dataset[0], dataset[1]


"""===================================DATA PREPROCESSING=============================="""


def data_preprocessing(x_train, x_test, augment):
    if augment > 0:
        shape = 14400, 1

        x_train_in, x_train_out = data_augmentation()

        x_train_in = [np.reshape(x, shape) for x in x_train_in]
        x_train_out = [np.reshape(x, shape) for x in x_train_out]
        
        x_train_in = np.array(x_train_in[:augment])
        x_train_out = np.array(x_train_out[:augment])
        
        x_train_in = np.concatenate((x_train[0], x_train_in), axis=0)
        x_train_out = np.concatenate((x_train[1], x_train_out), axis=0)

        x_train_in, x_train_out = normalize(x_train_in, x_train_out)
    else:
        x_train_in, x_train_out = normalize(x_train[0], x_train[1])

    x_test_in, x_test_out = normalize(x_test[0], x_test[1])

    if augment > 0:
        valid = int(len(x_train_in) * 0.2)
        x_valid_in = x_train_in[-valid:]
        x_valid_out = x_train_out[-valid:]

        x_train_in = x_train_in[:-valid]
        x_train_out = x_train_out[:-valid]
    else:
        valid = 0

    train_set = (x_train_in, x_train_out)
    test_set = (x_test_in, x_test_out)

    if augment > 0:
        validation = (x_valid_in, x_valid_out)

    valid_set = validation if augment > 0 else valid

    return train_set, test_set, valid_set


def normalize(input_d, output_d):
    height = 120
    width = 120

    x = np.reshape(input_d, (len(input_d), height, width, 1)).astype('float32') / 255
    y = np.reshape(output_d, (len(output_d), height, width, 1)).astype('float32') / 255
    return np.asarray(x), np.asarray(y)
