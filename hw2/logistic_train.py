import pandas as pd
import numpy as np
from util import *
import timeit

start = timeit.timeit()

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train'))
x_train = np.concatenate((x_train, np.ones((x_train.shape[0],1))), axis=1)

iter_num = 10000
lr = 5e-3
w = np.zeros((x_train.shape[1],1))
grad_accu = 0

for it in range(iter_num):
    f = sigmoid(np.dot(x_train, w))
    loss = np.sum(cross_entropy(y_train, f))
    w_grad = np.dot(x_train.T, f-y_train)
    grad_accu += w_grad**2
    lr_w = lr/np.sqrt(grad_accu/(it+1))
    w = w - lr_w*w_grad
    print('{0:.3f}\t{1:.3f}\t{2:d}'.format(loss, compute_accuracy(f,y_train), it))

pd.DataFrame(w).to_csv('para/logistic_para', index=False)

end = timeit.timeit()
