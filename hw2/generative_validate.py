import numpy as np
import pandas as pd
from util import *

# validation
x_va = np.array(pd.read_csv('data/proc_x_va'))
x_va = np.concatenate((x_va, np.ones((x_va.shape[0],1))), axis=1)
y_va = np.array(pd.read_csv('data/proc_y_va'))
w = np.array(pd.read_csv('para/generative_para')).T

ans = np.zeros(y_va.shape)
for i in range(x_va.shape[0]):
    z0 = np.dot(w[0], x_va[i])
    z1 = np.dot(w[1], x_va[i])
    if sigmoid(z1)>sigmoid(z0):
        ans[i] = 1

print(compute_accuracy(ans, y_va))