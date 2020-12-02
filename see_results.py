#!/usr/bin/env python3

import pickle
import numpy as np


f = open('iris_perceptron_C.pckl', 'rb')
iris_perceptron_C = pickle.load(f)
f.close()

f = open('iris_adaline_C.pckl', 'rb')
iris_adaline_C = pickle.load(f)
f.close()

for element in iris_perceptron_C:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], iris_perceptron_C[element]))

for element in iris_adaline_C:
    print("eta={}, k={}:\n C={}".format(element[0], element[1], iris_adaline_C[element]))