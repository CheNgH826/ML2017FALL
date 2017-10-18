import numpy as np
import pandas as pd
from util import *

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train'))

class1_size = np.sum(y_train)
class0_size = y_train.shape[0]-class1_size
print(class0_size, class1_size)
#class0_data = np.empty((class0_size, x_train.shape[1]))
#print(class0_data.shape)
#class1_data = np.empty((class1_size, x_train.shape[1]))
class0_data = []
class1_data = []

for i in range(y_train.shape[0]):
    if y_train[i]==0:
        class0_data.append(x_train[i])
    else:
        class1_data.append(x_train[i])

#print(len(class0_data), len(class1_data))
#print(class0_data[0])

class0_data = np.array(class0_data)
#print(class0_data.shape)
class1_data = np.array(class1_data)

avg0 = np.mean(class0_data, axis=0)
avg1 = np.mean(class1_data, axis=0)

cova1 = 
cova2 = 
