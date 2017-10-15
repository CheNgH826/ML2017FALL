import pandas as pd
import numpy as np
from util import *

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train'))
print(x_train.shape)

iter_num = 10000
lr = 1e-3
w = np.zeros((x_train.shape[1],1))
grad_accu = 0

for it in range(iter_num):
    f = sigmoid(np.dot(x_train, w))
    #f = f.reshape((x_train.shape[0],1))
    loss = np.sum(cross_entropy(y_train, f))
    w_grad = np.dot(x_train.T, f-y_train)
    grad_accu += w_grad**2
    lr_w = lr/np.sqrt(grad_accu/(it+1))
    w = w - lr_w*w_grad
    print('{0:.5f}\t{1:.5f}\t{2:d}'.format(loss, compute_accuracy(f,y_train), it))

pd.DataFrame(w).to_csv('paras/logistic_para', index=False)
