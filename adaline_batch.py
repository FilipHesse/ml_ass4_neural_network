#!/usr/bin/env python3
import numpy as np
from splitting import split_and_train
from testing import comp_error_rate

def adaline(data, eta, k=None):
    return split_and_train(adaline_train, data, eta, k)


def adaline_train(data, eta):
    x = data[:,:-1]
    t = data[:,-1]
    n = np.size(x,0)                    #number of inputs = nr of rows
    x = np.c_[np.ones(n), data[:,:-1]]  #Append column of ones at the beginning of all x for multiplication with w0
    d = np.size(x,1)                    #dimension of input = nr of columns
    w = 2*(np.random.rand(d)-0.5)       #Weights initialized randomly in range [-1 1]
                                        #Dimension is one higher than dimension of x because of constant bias
    w_last = w
    delta = 2*(np.ones(d))              #delta is array of 2s in the beginning
    l = 0                               #starting index
    iterations_over_dataset = 0
    stop = False
    sum = 0
    print("Perceptron_train: Start training ----------------------")
    while (not stop): #while delta (error) is not zero
        r = np.dot(w,x[l,:])
        a = np.sign(r)
        delta = (t[l]-r)
        sum += delta*x[l,:]
        l += 1
        if l==n:
            l=0
            stop =True

    dw = 1/n*sum
    w=w+dw
    return w