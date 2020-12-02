#!/usr/bin/env python3
import numpy as np

def comp_error_rate(x,w,t):
    a = np.sign(x@w)
    err_cnt = np.count_nonzero(~(a == t))
    return err_cnt/np.size(x,0)

def comp_confusion_matrix(data,w):
    """Computes confusion matrix for datasets with targets [1,any_number]
    """
    x = data[:,:-1]
    n = np.size(x,0)
    x = np.c_[np.ones(n), data[:,:-1]]
    t = data[:,-1]
    class1 = 1

    a = np.sign(x@w)

    C=np.array( [[np.count_nonzero((a==class1)[t==class1])/n , np.count_nonzero((~(a==class1))[t==class1])/n],
                [np.count_nonzero((a==class1)[~(t==class1)])/n , np.count_nonzero((~(a==class1))[~(t==class1)])/n]])

    return C