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
import pickle

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

eta = [0.2, 0.5, 0.8]
k = [2, 3, 4, 5, 10, 50, 100, 150]
params = [(x,y) for x in eta for y in k]

iris_C = {}
for param in params:
    iris_C[param] = perceptron(iris, *(param))

adaline(iris, 0.0001, 3)
#perceptron(#call entire matrix function! mnist.images_train, 0.002, 3)

print(C)

f = open('store.pckl', 'wb')
pickle.dump(obj, f)
f.close()

f = open('store.pckl', 'rb')
obj = pickle.load(f)
f.close()


# %%
