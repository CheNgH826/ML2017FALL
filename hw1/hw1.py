
import pandas as pd
import numpy as np

"""
y = wx + b
L = (y-y^)**2

"""

b = 0
w = np.ones((17, 1), dtype=np.float64)
data = pd.read_csv('train.csv', encoding='big5')

data_list = []
for j in data.index:
    one_day = []
    for i in data.columns:
        x1 = np.array(data[i][0+j*18:10+j*18])
        x2 = np.array(data[i][11+j*18:18+j*18])
        x = np.concatenate((x1,x2), axis=0)
        one_day.append(x)
    print(one_day)
    data_list.append(one_day)

