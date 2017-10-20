from sklearn.linear_model import SGDClassifier
import numpy as np
import pandas as pd
from util import *

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train'))
clf = SGDClassifier(loss='log', penalty='elasticnet')
clf.fit(x_train, y_train)

x_va = np.array(pd.read_csv('data/proc_x_va'))
y_va = np.array(pd.read_csv('data/proc_y_va'))
f = clf.predict(x_va)
print(np.sum(np.logical_not(np.logical_xor(f.reshape(-1,1),y_va)))/len(f))

acc_av = []
for i in range(50):
    clf.fit(x_train, y_train)
    f = clf.predict(x_va)
    acc = np.sum(np.logical_not(np.logical_xor(f.reshape(-1,1),y_va)))/len(f) 
    print(acc)
    acc_av.append(acc)

print(np.mean(acc_av))


