#!/usr/bin/env python3
import os
import numpy as np

def load_iris():    
    #specify full path
    fullpath = os.path.join(os.path.dirname(__file__),'data', 'iris-2class.txt')
    data = open(fullpath).read().split()
    return np.array(data, dtype=np.float32).reshape(-1,3)
