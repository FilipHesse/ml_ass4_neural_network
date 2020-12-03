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
train_iris = False
train_mnist = False
train_xor = True

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
        logging.info('Training iris with perceptron nr %d',i)
        iris_perceptron_C[param] = perceptron(iris, *(param))

    #Save computed confusion matrices
    f = open('iris_perceptron_C.pckl', 'wb')
    pickle.dump(iris_perceptron_C, f)
    f.close()

    #Train iris with adaline
    iris_adaline_C = {}
    for i, param in enumerate(params_adaline):
        logging.info('Training iris with adaline nr %d',i)
        iris_adaline_C[param] = adaline(iris, *(param))

    f = open('iris_adaline_C.pckl', 'wb')
    pickle.dump(iris_adaline_C, f)
    f.close()

    print(iris_perceptron_C)
    print(iris_adaline_C)

if train_mnist:
    # Load data
    mnist=MnistData()

    # for i in np.arange(20):
    #     print(mnist.isolate_class_from_trainset(2)[i,-1])
    #     plt.imshow(mnist.isolate_class_from_trainset(2)[i,:-1].reshape(28,28))
    #     plt.show()

    #Configute parameters
    eta_perceptron = [0.2]
    eta_adaline = [0.0000001]
    k = [2,3]
    params_perceptron = [(x,y) for x in eta_perceptron for y in k]
    params_adaline = [(x,y) for x in eta_adaline for y in k]

    #test1 = perceptron(mnist.isolate_class_from_trainset(1)[:100], 0.2, 2)
    test2 = adaline(mnist.isolate_class_from_trainset(1)[:1000], 0.0000001, 2)

    #Train mnist with perceptron
    mnist_perceptron_C_1 = {}
    mnist_perceptron_C_2 = {}
    for i, param in enumerate(params_perceptron):
        logging.info('Training mnist digit1 with perceptron. eta={} k={}'.format(param[0], param[1]))
        mnist_perceptron_C_1[param] = perceptron(mnist.isolate_class_from_trainset(1)[:1000], *(param))
        logging.info('Training mnist digit2 with perceptron. eta={} k={}'.format(param[0], param[1]))
        mnist_perceptron_C_2[param] = perceptron(mnist.isolate_class_from_trainset(2)[:1000], *(param))
            
        #Save computed confusion matrices
        f = open('mnist_perceptron_C_1.pckl', 'wb')
        pickle.dump(mnist_perceptron_C_1, f)
        f.close()

        #Save computed confusion matrices
        f = open('mnist_perceptron_C_2.pckl', 'wb')
        pickle.dump(mnist_perceptron_C_2, f)
        f.close()

    #Train mnist with adaline
    mnist_adaline_C_1 = {}
    mnist_adaline_C_2 = {}
    for i, param in enumerate(params_adaline):
        logging.info('Training mnist digit1 with adaline. eta={} k={}'.format(param[0], param[1]))
        mnist_adaline_C_1[param] = adaline(mnist.isolate_class_from_trainset(1)[:1000], *(param))
        logging.info('Training mnist digit2 with adaline. eta={} k={}'.format(param[0], param[1]))
        mnist_adaline_C_2[param] = adaline(mnist.isolate_class_from_trainset(2)[:1000], *(param))


        #Save computed confusion matrices
        f = open('mnist_adaline_C_1.pckl', 'wb')
        pickle.dump(mnist_adaline_C_1, f)
        f.close()

        #Save computed confusion matrices
        f = open('mnist_adaline_C_2.pckl', 'wb')
        pickle.dump(mnist_adaline_C_2, f)
        f.close()

if train_xor:
    # Load
    xor=np.array([[0, 0, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0],
                    [0, 0, 0],
                    [1, 0, 1],
                    [0, 1, 1],
                    [1, 1, 0]])

    # class1 = xor[xor[:,2]==0,:]
    # class2 = xor[xor[:,2]==1,:]
    # plt.plot(class1[:,0], class1[:,1], "bs")
    # plt.plot(class2[:,0], class2[:,1], "g^")
    # plt.show()

    #Configute parameters
    eta_perceptron = [0.5, 0.1, 0.01, 0.001]
    eta_adaline = [0.5, 0.1, 0.01, 0.001]
    k = [2]
    params_perceptron = [(x,y) for x in eta_perceptron for y in k]
    params_adaline = [(x,y) for x in eta_adaline for y in k]

    #Train xor with perceptron
    xor_perceptron_C = {}
    for i, param in enumerate(params_perceptron):
        logging.info('Training xor with perceptron. eta={} k={}'.format(param[0], param[1]))
        xor_perceptron_C[param] = perceptron(xor, *(param))

    #Save computed confusion matrices
    f = open('xor_perceptron_C.pckl', 'wb')
    pickle.dump(xor_perceptron_C, f)
    f.close()

    #Train xor with adaline
    xor_adaline_C = {}
    for i, param in enumerate(params_adaline):
        logging.info('Training xor with adaline. eta={} k={}'.format(param[0], param[1]))
        xor_adaline_C[param] = adaline(xor, *(param))

    f = open('xor_adaline_C.pckl', 'wb')
    pickle.dump(xor_adaline_C, f)
    f.close()

    print(xor_perceptron_C)
    print(xor_adaline_C)