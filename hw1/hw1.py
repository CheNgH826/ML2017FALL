
import pandas as pd
import numpy as np
from random import randint


# data reading
data = pd.read_csv('train.csv', encoding='big5')
data_list = []
for j in range(int(len(data.index)/18)):
    one_day = []
    for i in range(24):
        x = np.array(data[str(i)][0+j*18:18+j*18])
        if x[10]=='NR':
            x[10]='0'
        one_day.append(x.astype(float))
    data_list.append(one_day)
data_list = np.array(data_list)

# learning
# y = wx + b
# L = (y-y^)**2
# data_list.shape=(240, 24, 18)(days, hours, items)
b = 0.0
w = np.ones((18*9,), dtype=np.float64)
day_num = int(data_list.shape[0]/2)  # half training half validation
loss = 0.0

w_grad = np.zeros((18*9,), dtype=float)
b_grad = 0.0

# compute loss
for date in range(day_num): # 120
    for hour in range(data_list.shape[1]-8): # 16
        x = np.empty((0))
        for i in range(hour, hour+9):
            x = np.append(x, data_list[date][hour])
            #print(data_list[date][hour].shape)
            #np.concatenate(x, data_list[date][hour])
        # print(x.shape)
        # x.reshape((1, 18*9))
        y_ans = data_list[date][hour][9] # PM2.5
        y_est = np.dot(x, w)+b
        loss += (y_ans-y_est)**2

        b_grad += -2*(y_ans-y_est)
        w_grad += 2*(y_ans-y_est)*np.transpose(x)

print(loss)

# gradient decent
l_rate = 0.1

w = w-l_rate*w_grad
b = b-l_rate*b_grad
# print(w)
# print(b)

for date in range(day_num): # 120
    for hour in range(data_list.shape[1]-8): # 16
        x = np.empty((0))
        for i in range(hour, hour+9):
            x = np.append(x, data_list[date][hour])
            #print(data_list[date][hour].shape)
            #np.concatenate(x, data_list[date][hour])
        # print(x.shape)
        # x.reshape((1, 18*9))
        y_ans = data_list[date][hour][9] # PM2.5
        y_est = np.dot(x, w)+b
        loss += (y_ans-y_est)**2

        b_grad += -2*(y_ans-y_est)
        w_grad += 2*(y_ans-y_est)*np.transpose(x)

print(loss)
