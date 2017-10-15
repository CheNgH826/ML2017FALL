import pandas as pd
import numpy as np
from util import *

x_va = np.array(pd.read_csv('data/proc_x_va'))
y_va = np.array(pd.read_csv('data/proc_y_va'))
w = pd.read_csv('paras/logistic_para')

f = sigmoid(np.dot(x_va, w))
loss = np.sum(cross_entropy(y_va, f))
print(loss)
print(compute_accuracy(f,y_va))
