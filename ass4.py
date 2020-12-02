#!/usr/bin/env python3

# To use script with the correct data, please download the given MNIST dataset (aulaweb version)
# and extract the files inside the "data" directory. The matlab functions are not needed.
# %%
from mnist_data import MnistData 
import os
import numpy as np
import matplotlib.pyplot as plt
from perceptron import perceptron
from adaline import adaline
from iris_data import load_iris

# Load data
mnist=MnistData()
mnist_trainset_1 = mnist.isolate_class_from_trainset(1)

iris=load_iris()
# class1 = iris[iris[:,2]==-1,:]
# class2 = iris[iris[:,2]==1,:]
# plt.plot(class1[:,0], class1[:,1], "bs")
# plt.plot(class2[:,0], class2[:,1], "g^")
# plt.show()
# %%
perceptron(iris, 0.5, 150)
# C = [perceptron(iris, 0.5, 2), 
#     perceptron(iris, 0.5, 3),
#     perceptron(iris, 0.5, 10),
#     perceptron(iris, 0.5, 100)]
adaline(iris, 0.0001, 3)
#perceptron(#call entire matrix function! mnist.images_train, 0.002, 3)

print(C)


# %%
