import numpy as np
import pandas as pd
from util import *

def likelihood(avg, cova, x):
    inv_cova = np.linalg.inv(cova)
    # print(np.linalg.det(cova))
    const = ((2*np.pi)**cova.shape[0]*np.linalg.det(cova))**(-0.5)
    exp_power = -0.5*np.dot(np.dot(x-avg, np.linalg.inv(cova)), (x-avg).T)
    return const*np.exp(exp_power)

def posterier(avg, cova, x, type_prob, avg_the_other):
    prob_x_and_type = likelihood(avg, cova, x)*type_prob
    prob_x = prob_x_and_type + likelihood(avg_the_other, cova, x)*(1-type_prob)
    return prob_x_and_type/prob_x

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train'))

class1_size = np.sum(y_train)
class0_size = y_train.shape[0]-class1_size
# print(class0_size, class1_size)
prob_0 = float(class0_size/y_train.shape[0])
prob_1 = 1-prob_0
class0_data = []
class1_data = []

for i in range(y_train.shape[0]):
    if y_train[i]==0:
        class0_data.append(x_train[i])
    else:
        class1_data.append(x_train[i])

class0_data = np.array(class0_data)
class1_data = np.array(class1_data)

avg0 = np.mean(class0_data, axis=0)
avg1 = np.mean(class1_data, axis=0)

cova0 = np.dot((class0_data-avg0).T, class0_data-avg0)/class0_data.shape[0]
cova1 = np.dot((class1_data-avg0).T, class1_data-avg0)/class1_data.shape[0]
# print(cova1.shape)
cova = cova0*prob_0+cova1*prob_1


# validation
x_va = np.array(pd.read_csv('data/proc_x_va'))
y_va = np.array(pd.read_csv('data/proc_y_va'))
ans = np.zeros(y_va.shape)

for i in range(x_va.shape[0]):
    if posterier(avg0, cova, x_va[i], prob_0, avg1) < posterier(avg1, cova, x_va[i], prob_0, avg0):
        ans[i]=1

print(compute_accuracy(ans, y_va))