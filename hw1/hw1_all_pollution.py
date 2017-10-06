import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
num_day=240
num_hour=24

# data reading
print('data reading and model building...')
data = pd.read_csv('train.csv', encoding='big5')
x = np.empty((0, 18*9), float)
y = []
x_va = np.empty((0, 18*9), float)
y_va = []
for day in range(num_day):
    for h in range(num_hour):
        if data[str(h)][10+day*18] == 'NR':
            data[str(h)][10+day*18] = '0'

    for hour in range(num_hour-9):
        xi = np.array([])
        for idx in range(1, 10):
            xi = np.concatenate((xi, data[str(hour+idx)][day*18:day*18+18]), axis=0)
        if day < num_day*3/4:
            x = np.vstack((x, xi))        
            y.append(data[str(hour+9)][9+day*18])
        else:
            x_va = np.vstack((x_va, xi))        
            y_va.append(data[str(hour+9)][9+day*18])


b = np.zeros(int(num_day*(num_hour-9)*3/4), )
w = np.random.rand(18*9, )

print('training...')
iter_num = 100000
lr = 0.01
lr_w = np.random.rand(18*9,)
lr_b = 0.
loss_history = []

x = np.array(x, dtype=float)
y = np.array(y, dtype=float)

for it in range(iter_num):
    y_hat = np.dot(x, w)+b
    diff = y_hat-y
    loss = np.sqrt(np.mean(np.square(diff)))
    b_grad = 2*np.sum(diff)
    w_grad = 2*np.dot(x.T, diff)

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
b_va = b[:900]
y_hat = np.dot(x_va, w)+b_va
diff_va = y_hat-y_va
loss_va = np.sqrt(np.mean(np.square(diff_va)))
print('validation loss: ', loss_va)

b_val = b[0]
print('generating answers...')
test_data = pd.read_csv('test.csv', header=None, encoding='big5')
num_day = test_data.shape[0]/18
num_hour = 9
y_ans=[]
for day in range(int(num_day)):
    x = np.empty((0, 18*9), float)
    for h in range(num_hour):
        if test_data[h+2][10+day*18] == 'NR':
            test_data[h+2][10+day*18] = '0'

    xi = np.array([])
    for hour in range(num_hour):
        xi = np.concatenate((xi, test_data[hour+2][day*18:day*18+18]), axis=0)
    x = np.vstack((x, xi))
    # print(b.shape, x.shape, w.shape)
    x = np.array(x, dtype=float)
    y_hat = np.dot(x, w)+b_val
    # print(y_hat)
    y_ans.append(y_hat[0])

id_array = []
for day in range(int(num_day)):
    id_array.append('id_'+str(day))
# print(id_array)
# print(y_ans)
ans_np = np.array([id_array, y_ans])
ans_dataframe = pd.DataFrame(ans_np.T)
ans_dataframe.to_csv('ans_all_pollution.csv', index=False, header=['id','value'])

print(loss_history[-10:-1])
plt.plot(loss_history)
plt.yscale('log')
plt.show()
