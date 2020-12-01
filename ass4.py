#!/usr/bin/env python3

# To use script with the correct data, please download the given MNIST dataset (aulaweb version)
# and extract the files inside the "data" directory. The matlab functions are not needed.
from mnist_data import MnistData 
import os
import numpy as np
from perceptron import perceptron
from adaline import adaline
from iris_data import load_iris

#Load data
mnist=MnistData()
iris=load_iris()


mnist.get_train_set_one_matrix()
mnist.get_train_set_one_matrix(1)
mnist.get_train_set_one_matrix_complement(1)
perceptron(mnist.images_train, 0.002, 3)

pass

