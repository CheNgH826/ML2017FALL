import pandas as pd
import numpy as np
from scipy.special import expit

def standardize(matrix): # matrix.shape = (m, n)
    avg = np.mean(matrix, axis=0)
    print(avg.shape)
    print(matrix.shape)
    std = np.std(matrix, axis=0)
    return (matrix-avg)/std

def cross_entropy(p,q):
    return -(p*np.log(q)+(1-p)*np.log(1-q)).astype(float)

def sigmoid(x):
    # return 1/(1+np.exp(-x))
    return expit(x)

def determine_ans(f):
    ans = np.zeros(f.shape)
    for i in range(len(f)):
        if f[i] >= 0.5:
            ans[i]=1
    return ans.astype(int)


def compute_accuracy(f,y):
    ans = determine_ans(f)
    return np.sum(np.logical_not(np.logical_xor(ans,y)))/len(f)
