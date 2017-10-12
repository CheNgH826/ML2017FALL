import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import sys

num_day=240
num_hour=24
feat_hour=5

print('data reading and model building...')
data = pd.read_csv('train.csv', encoding='big5')
x = []
y = []
x_va = []
y_va = []

x_month=[]
for month in range(12):
    xm = []
    for day in range(20):
        for hour in range(24):
            xm.append(data[str(hour)][day*18+9+month*18*20])
    
    if month < 9:
        for n in range(24*20-feat_hour):
            x.append(xm[n:n+feat_hour])
            y.append(xm[n+feat_hour])
    else:
        for n in range(24*20-feat_hour):
            x_va.append(xm[n:n+feat_hour])
            y_va.append(xm[n+feat_hour])
#print(len(xm))

x = np.array(x, dtype=float)
x_va = np.array(x_va, dtype=float)
y = np.array(y, dtype=float)

#b = np.zeros(int(num_day*(num_hour-9)*3/4), )
b = np.zeros(y.shape[0])
w = np.zeros(feat_hour, )
w *= 0.01

print('training...')
iter_num = 100000
lr = 1e-3
lr_w = np.zeros(feat_hour,)
lr_b = 0.
loss_history = []

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)
lumbda = 0.0001
for it in range(iter_num):
    y_hat = np.dot(x, w)+b
    diff = y_hat-y
    #loss_i = np.square(diff)+lumbda*np.square(w)
    loss = np.sqrt(np.mean(np.square(diff)))
    b_grad = 2*np.sum(diff)
    w_grad = 2*np.dot(x.T, diff)+2*lumbda*w

    lr_w += np.square(w_grad)
    lr_b += b_grad**2
    w = w - lr*w_grad/np.sqrt(lr_w/(it+1))
    b = b - lr*b_grad/np.sqrt(lr_b/(it+1))
    #w = w - lr*w_grad
    #b = b - lr*b_grad

    loss_history.append(loss)

print('validating...')
x_va = np.array(x_va, dtype=float)
y_va = np.array(y_va, dtype=float)
#b = b[:int(num_day*(num_hour-9)/4)]
b_va = b[:y_va.shape[0]]
y_hat = np.dot(x_va, w)+b_va
diff = y_hat-y_va
loss_va = np.sqrt(np.mean(np.square(diff)))
print('validation loss: ', loss_va)


b_val = b[0]
print('generating answers...')
test_data = pd.read_csv('test.csv', header=None, encoding='big5')
num_day = test_data.shape[0]/18
num_hour = 9
y_ans=[]
for day in range(int(num_day)):
    x = []
    for h in range(num_hour):
        if test_data[h+2][10+day*18] == 'NR':
            test_data[h+2][10+day*18] = '0'

    xi = []
    for hour in range(num_hour-feat_hour, num_hour):
        xi.append(test_data[hour+2][day*18+9])
    x.append(xi)
    x = np.array(x, dtype=float)
    y_hat = np.dot(x, w)+b_val
    y_ans.append(y_hat[0])

id_array = []
for day in range(int(num_day)):
    id_array.append('id_'+str(day))
ans_np = np.array([id_array, y_ans])
ans_dataframe = pd.DataFrame(ans_np.T)
#ans_dataframe.to_csv('ans_pm25_regu.csv', index=False, header=['id','value'])
ans_dataframe.to_csv(sys.argv[1], index=False, header=['id','value'])

print(loss_history[-10:])
plt.plot(loss_history)
plt.yscale('log')
plt.show()
