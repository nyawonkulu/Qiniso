# -*- coding: utf-8 -*-

import _pickle as pickle
import gzip
from glob import glob
import PIL
from PIL import Image
import numpy as np
import math
import matplotlib.pyplot as plt

def dir_to_dataset(glob_files):

    print('\n\n======================================================================================\n\n')
    print('%35s PROCESSING ' % (''))
    print('\n\n======================================================================================\n')
    dataset = []
    files = 0

    for file_count, file_name in enumerate(sorted(glob(glob_files), key=len)):
        # tograyscale
        img = Image.open(file_name).convert('LA')

        #exit(0)
        if img.size != (120, 120):
            img = img.resize((120, 120), Image.NEAREST)

        pixels = [f[0] for f in list(img.getdata())]

        dataset.append(pixels)
        print('%34s %s files processed' % ('', files))
        files += 1

    return np.array(dataset), files


def partition(Data, file_count):
    trn_size = int(math.ceil(file_count * 0.90))  # train data size
    test_size = int(math.ceil(file_count * 0.10))  # test data size

    test_idx = 1 + trn_size
    data_size = trn_size + test_size  # Total size of data
    # Data and labels are created based on the specific number

    train_in = Data[:trn_size]
    test_in = Data[test_idx:data_size]
    return train_in, test_in



def multiple_datasets(glob_files, train_in, train_out, test_in, test_out):
    set_2 = ['Dataset 1']
    #, 'Dataset 2']

    for k in set_2:
        Data, file_count = dir_to_dataset(glob_files + '/' + k + '/*.bmp')
        t_in, tt_in = partition(Data, file_count)

        Data, file_count = dir_to_dataset(glob_files + '/' + k + '/*.png')
        t_out, tt_out = partition(Data, file_count)

        train_in.extend( t_in)
        train_out.extend(t_out)

        test_in.extend(tt_in)
        test_out.extend(tt_out)

# def dir_to_dataset(glob_files, loc_train_labels=""):
def dataset(glob_files):
    # start write pkl.gz file
    f = gzip.open('./dataset/file_out.pkl[90].gz', 'wb')

    train_in,test_in = [], []
    train_out, test_out = [], []


    Data, file_count = dir_to_dataset(glob_files + '/*.bmp')
    train_in, test_in = partition(Data, file_count)

    Data, file_count = dir_to_dataset(glob_files + '/*.png')
    train_out, test_out = partition(Data, file_count)

    print('\n\n\n======================================================================================\n')

    train_set_x2 = []
    train_set_y2 = []
    test_set_x2 = []
    test_set_y2 = []

    # shuffle train, val, test data, respectively

    arr = np.arange(len(train_in))
    np.random.shuffle(arr)

    for i in arr:
        train_set_x2.append(train_in[i])
        train_set_y2.append(train_out[i])

    print('TRAINING SHUFFLING COMPLETED....')

    arr = np.arange(len(test_in))
    np.random.shuffle(arr)

    for i in arr:
        test_set_x2.append(test_in[i])
        test_set_y2.append(test_out[i])

    print('TEST SHUFFLING COMPLETED....')

    print(len(train_in), len(test_in))

    train_set = (np.array(train_set_x2), np.array(train_set_y2))
    test_set = (np.array(test_set_x2), np.array(test_set_y2))

    dumping = [train_set, test_set]
    print("DATASET MERGED")

    print('DUMPING....')
    pickle.dump(dumping, f, protocol=2)

    print('\n====================================== COMPLETED =====================================\n')
    f.close()


def display_():
    f = gzip.open('./dataset/file_out.pkl[90].gz', 'rb')

    training_data, test_data = pickle.load(f, encoding='latin1')

    print('\n======================================================================================\n')
    print('%20s %s %22s ' % ('', 'IMAGE INTENSITY VALUES', 'IMAGE LABEL'))

    print('\n======================================================================================\n')

    plt.figure(figsize=(300, 300))
    n = 20

    for image in range(n):
        print('%16s %s %10s ' % (
        '', str(test_data[0][image]), str(test_data[1][image])))
        plt.imshow(test_data[0][image].reshape(128, 128))

        ax = plt.subplot(2, n, image + 1)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        plt.imshow(test_data[1][image].reshape(120, 120))
        ax = plt.subplot(2, n, image + 1 + n)
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    f.close()
    plt.show()


if __name__ == '__main__':
    print('\n======================================================================================\n')
    process = int(
        input('%17s GENERATE A NEW DATASET [0] USE EXISTING DATASET [1]: ' % ('')))
    print('\n\n======================================================================================\n\n')

    # noinspection PyBroadException
    # resize('C:/Users/213539359/Documents/Project/dataset/LISC Database/LISC Database/Main Dataset/')
    #dataset('C:/Users/213539359/Documents/Project/dataset/LISC Database/LISC Database/Main Dataset/')

    if process == 0:
        dataset('C:/Users/213539359/COMP/COMP700 - Honours Project/Project/Document Cloud/dataset/segmentation_WBC-master/segmentation_WBC-master/Dataset 1')
    else:
        display_()
