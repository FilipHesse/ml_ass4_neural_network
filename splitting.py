#!/usr/bin/env python3
import numpy as np
from testing import comp_confusion_matrix

def split_and_train(train, data, eta, k=None):
    if eta <= 0 or eta >1:
        raise Exception("Eta should be in the range ]0, 1]")

    if k < 2 or k > np.size(data, 0):
        raise Exception("k should be in the range [2, {}]".format(np.size(data, 0)))

    total_length = np.size(data, 0)

    if k >= 2 and k <= np.size(data, 0):
        # perform k-fold cross validation
        total_length = np.size(data, 0)
        perm = np.random.permutation(total_length)
        S = np.empty((k,2,2))
        for i in np.arange(0, k):
            # divide whole dataset into k subsets and in the loop take always another one of them
            # as the testset
            boolean_mask = np.full(total_length, False)
            boolean_mask[int(i*total_length/k + 0.5):int((i+1)*total_length/k + 0.5)] = True

            training_set = data[perm,:][~boolean_mask]
            test_set = data[perm,:][boolean_mask]
            w = train(training_set, eta)
            #compute confusion Matrix C, Add R(xi) to set S
            S[i,:,:] = comp_confusion_matrix(test_set,w)
        pass
        #Train over total data --> not needed in this context, we only want to know confusion Matrix
        #compute average confusion matrix
        return np.sum(S,0)/k

