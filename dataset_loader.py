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

"""===================================LOAD DATA==============================="""


def load_data():
    """Return the MNIST data as a tuple containing the training data,
    the validation data, and the test data.

    The ``training_data`` is returned as a tuple with two entries.
    The first entry contains the actual training images.  This is a
    numpy ndarray with 50,000 entries.  Each entry is, in turn, a
    numpy ndarray with 784 values, representing the 28 * 28 = 784
    pixels in a single MNIST image.

    The second entry in the ``training_data`` tuple is a numpy ndarray
    containing 50,000 entries.  Those entries are just the digit
    values (0...9) for the corresponding images contained in the first
    entry of the tuple.

    The ``validation_data`` and ``test_data`` are similar, except
    each contains only 10,000 images.

    This is a nice data format, but for use in neural networks it's
    helpful to modify the format of the ``training_data`` a little.
    That's done in the wrapper function ``load_data_wrapper()``, see
    below.
    """
    #f = gzip.open('mnist.pkl.gz', 'rb')
    f = gzip.open('./dataset/file_out.pkl[90].gz', 'rb')
    training_data, test_data = pickle.load(f,  encoding='latin1')
    f.close()
    return training_data, test_data

"""===================================LOAD DATA WRAPPER==============================="""


def load_data_wrapper():
    """Return a tuple containing ``(training_data, validation_data,
    test_data)``. Based on ``load_data``, but the format is more
    convenient for use in our implementation of neural networks.

    In particular, ``training_data`` is a list containing 50,000
    2-tuples ``(x, y)``.  ``x`` is a 784-dimensional numpy.ndarray
    containing the input image.  ``y`` is a 10-dimensional
    numpy.ndarray representing the unit vector corresponding to the
    correct digit for ``x``.

    ``validation_data`` and ``test_data`` are lists containing 10,000
    2-tuples ``(x, y)``.  In each case, ``x`` is a 784-dimensional
    numpy.ndarry containing the input image, and ``y`` is the
    corresponding classification, i.e., the digit values (integers)
    corresponding to ``x``.

    Obviously, this means we're using slightly different formats for
    the training data and the validation / test data.  These formats
    turn out to be the most convenient for use in our neural network
    code."""
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



"""
def gen():
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1./255)

    train_generator = train_datagen.flow_from_directory(
            'data/train',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    validation_generator = test_datagen.flow_from_directory(
            'data/validation',
            target_size=(150, 150),
            batch_size=32,
            class_mode='binary')

    model.fit_generator(
            train_generator,
            steps_per_epoch=2000,
            epochs=50,
            validation_data=validation_generator,
            validation_steps=800)
"""

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

    x_train_in = np.reshape(x_train_in, (x_train_in.shape[0], 120, 120, 1)).astype('float32') / 255
    x_train_out = np.reshape(x_train_out, (x_train_out.shape[0], 120, 120, 1)).astype('float32') / 255

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


def data_preprocessing(x_train, x_test, augment):
        height = 120
        width = 120

        if augment > 0:
            x_train_in, x_train_out = data_augmentation(x_train[0], x_train[1], augment)
        else:
            x_train_in = np.reshape(x_train[0], (x_train[0].shape[0], height, width, 1)).astype('float32') / 255
            x_train_out = np.reshape(x_train[1] ,(x_train[1].shape[0], height, width, 1)).astype('float32') / 255

        x_test_in = np.reshape(x_test[0], (x_test[0].shape[0], height, width, 1)).astype('float32') / 255
        x_test_out = np.reshape(x_test[1], (x_test[1].shape[0], height, width, 1)).astype('float32') / 255

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
