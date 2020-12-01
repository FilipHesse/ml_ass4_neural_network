#!/usr/bin/env python3
import numpy as np

def perceptron(data, eta, k=None):
    if eta <= 0 or eta >1:
        raise Exception("Eta should be in the range ]0, 1]")

    if k < 2 or k > np.size(data, 0):
        raise Exception("k should be in the range [2, {}]".format(np.size(data, 0)))

    total_length = np.size(data, 0)

    if k == 2:
        # split test set into two sets of equal size
        training_set_length = int( total_length/2 )
        perm = np.random.permutation(total_length)
        training_set = perm[:training_set_length]
        test_set = perm[training_set_length:]

    if k == np.size(data, 0):
        # perform leave one out cross validation
        for i in np.arange(0, total_length):
            training_set = np.delete(data, i, 0)
            test_set = data(i)
            #train(training_set)
            #compute confusion Matrix C
            #Add R(xi) to set S
        #Train over total data --> not needed in this context, we only want to know confusion Matrix
        #compute average confusion matrix

    if k > 2 and k < np.size(data, 0):
        # perform k-fold cross validation
        total_length = np.size(data, 0)
        training_set_length = int( total_length/k + 0.5) # + 0.5 for proper rounding
        perm = np.random.permutation(total_length)
        for i in np.arange(0, k):
            # divide whole dataset into k subsets and in the look take always another one of them
            # as the testset
            boolean_mask = np.full(total_length, False)
            if i < k-1:
                boolean_mask[i*training_set_length:(i+1)*training_set_length] = True
            else: #special treatment needed due to integer divisibility of set 
                boolean_mask[i*training_set_length:] = True #Till the end
            training_set = data[perm,:][~boolean_mask]
            test_set = data[perm,:][boolean_mask]
            #train(training_set)
            #compute confusion Matrix C
            #Add R(xi) to set S
        #Train over total data --> not needed in this context, we only want to know confusion Matrix
        #compute average confusion matrix


    if k == None:
        pass
        #Compute the confusion matrix on the training set

    # some comment about choosing the random value each time again => yes!
    pass

def perceptron_learn(data, eta):
    x = data[:-1]
    t = data[-1]
    d = np.size(x,1)    #dimension of input = nr of columns
    n = np.size(x,0)    #number of inputs = nr of rows
    w = 2*(np.random.rand(1,d)-0.5)       #weights initialized randomly in range [-1 1]
    l = 0               #starting index
    while (error):
        r = w*x[l,:]
        a = np.sign(r)
        delta = .5*(t[l]-a)
        dw = eta*delta*x[l,:]
        w = w + dw
        l += 1
        if l==n:
            l=0
            if not np.any(delta):   #delta (error) is zero
                print("The network converged")
                return w



