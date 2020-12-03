#!/usr/bin/env python3

import pickle
import numpy as np
import matplotlib.pyplot as plt

class TestResult:
    def __init__(self,eta,k,C):
        self.C = C
        self.eta = eta
        self.k = k
        self.compute_specificity()
        self.compute_sensitivity()
        self.compute_precision()
        self.compute_F_measure()
    
    def compute_sensitivity(self):
        self.sens=self.C[1,1]/(self.C[1,1] + self.C[1,0])

    def compute_specificity(self):
        self.spec=self.C[0,0]/(self.C[0,0] + self.C[0,1])

    def compute_precision(self):
        self.prec=self.C[1,1]/(self.C[1,1] + self.C[1,0])

    def compute_F_measure(self):
        self.F = 2*(self.prec*self.sens)/(self.prec+self.sens)

    
names = ['iris_perceptron_C','iris_adaline_C','mnist_perceptron_C_1', 'mnist_perceptron_C_2']

for name in names:
    f = open( name+ '.pckl', 'rb')
    C_dict = pickle.load(f)
    f.close()

    test_result=[]
    for element in C_dict:
        print("eta={}, k={}:\n C={}".format(element[0], element[1], C_dict[element]))
        test_result.append(TestResult(element[0], element[1], C_dict[element]))


    plt.plot([elem.k for elem in test_result],[elem.F for elem in test_result],"*")
    plt.title(name)
    plt.show() 



