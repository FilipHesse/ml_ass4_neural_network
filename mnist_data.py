#!/usr/bin/env python3
from mnist import MNIST
import os
import numpy as np

class MnistData():
    def __init__(self):
        data_path = os.path.join(os.path.dirname(__file__), 'data')

        # Some extraction methods result in wrong names for imported MNIST package
        # rename them
        wrong_names = [
            os.path.join(data_path, 't10k-images.idx3-ubyte'),
            os.path.join(data_path, 't10k-labels.idx1-ubyte'),
            os.path.join(data_path, 'train-images.idx3-ubyte'),
            os.path.join(data_path, 'train-labels.idx1-ubyte'),
        ]

        correct_names = [
            os.path.join(data_path, 't10k-images-idx3-ubyte'),
            os.path.join(data_path, 't10k-labels-idx1-ubyte'),
            os.path.join(data_path, 'train-images-idx3-ubyte'),
            os.path.join(data_path, 'train-labels-idx1-ubyte'),
        ]

        for i, wrong_name in enumerate(wrong_names):
            if os.path.isfile(wrong_name):
                os.rename(wrong_name,correct_names[i])

        mndata = MNIST(os.path.join(os.path.dirname(__file__), 'data'))

        images_train_list, labels_train_list = mndata.load_training()
        images_test_list, labels_test_list = mndata.load_testing()

        #Convert to np.array
        self.images_train = np.array(images_train_list)
        self.labels_train = np.array(labels_train_list)
        self.images_test = np.array(images_test_list)
        self.labels_test = np.array(labels_test_list)
    
    def get_subset_train(self,num):
        return self.images_train[self.labels_train == num,:]

    def get_subset_train_complement(self,num):
        return self.images_train[np.logical_not(self.labels_train == num),:]

    def get_subset_test(self,num):
        return self.images_test[self.labels_test == num,:]

    def get_subset_test_complement(self,num):
        return self.images_test[np.logical_not(self.labels_test == num),:]