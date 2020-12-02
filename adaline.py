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
    print("Perceptron_train: Start training ----------------------")
    while (not stop): #while delta (error) is not zero
        r = np.dot(w,x[l,:])
        a = np.sign(r)
        delta = (t[l]-r)
        dw = eta*delta*x[l,:]
        w = w + dw
        if iterations_over_dataset == 0:
            print("dw={}, w={}, x_{}".format(dw, w, l))
        l += 1
        if l==n:
            l=0
            iterations_over_dataset += 1
            #stop conditions
            error_rate = comp_error_rate(x,w,t)  #stop, if output exactly eqal target vector
            print("Error_rate= {}, w={}, iterations= {}".format(error_rate, w, iterations_over_dataset))
            if error_rate == 0:
                stop = True
                print("Perceptron_train: The network converged, error = 0-----------")
            elif np.all(w == w_last):
                stop = True
                print("Perceptron_train: w did not change anymore, stopping----------")
            #if iterations_over_dataset % 100 == 0:
            elif iterations_over_dataset == 100000:
                stop = True
                print("Perceptron_train: Stops without convergence: 100000 iterations over whole dataset completed")
            w_last = w    
    return w