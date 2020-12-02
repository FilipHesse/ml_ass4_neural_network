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
from configure_logging import configure_logging
import pickle
import logging

configure_logging()

# Tell the script what to do
display_iris_set = False
train_iris = True

# Load data
mnist=MnistData()
mnist_trainset_1 = mnist.isolate_class_from_trainset(1)

if train_iris:
    # Load
    iris=load_iris()

    #Display set
    if display_iris_set:
        class1 = iris[iris[:,2]==-1,:]
        class2 = iris[iris[:,2]==1,:]
        plt.plot(class1[:,0], class1[:,1], "bs")
        plt.plot(class2[:,0], class2[:,1], "g^")
        plt.show()

    #Configute parameters
    eta_perceptron = [0.2, 0.5, 0.8]
    eta_adaline = [0.0001, 0.0005, 0.001]
    k = [2, 3, 4, 5, 10, 50, 100, 150]
    params_perceptron = [(x,y) for x in eta_perceptron for y in k]
    params_adaline = [(x,y) for x in eta_adaline for y in k]

    #Train iris with perceptron
    iris_perceptron_C = {}
    for i, param in enumerate(params_perceptron):
        logging.info('Training with perceptron nr %d',i)
        iris_perceptron_C[param] = perceptron(iris, *(param))

    #Train iris with adaline
    iris_adaline_C = {}
    for i, param in enumerate(params_adaline):
        logging.info('Training with adaline nr %d',i)
        iris_adaline_C[param] = adaline(iris, *(param))

    #Save computed confusion matrices
    f = open('iris_perceptron_C.pckl', 'wb')
    pickle.dump(iris_perceptron_C, f)
    f.close()

    f = open('iris_adaline_C.pckl', 'wb')
    pickle.dump(iris_adaline_C, f)
    f.close()

    print(iris_perceptron_C)
    print(iris_adaline_C)

