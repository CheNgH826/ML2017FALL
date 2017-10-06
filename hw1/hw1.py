
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

num_day=240
num_hour=24

# data reading
data = pd.read_csv('train.csv', encoding='big5')
pm10_all = []
pm25_all = []
for day in range(num_day):
    pm10_one_day = []
    pm25_one_day = []
    for hour in range(num_hour):
        pm10_one_day.append(data[str(hour)][8+day*18])
        pm25_one_day.append(data[str(hour)][9+day*18])
    #print(pm10_one_day)
    #print(pm25_one_day)
    pm10_all.append(pm10_one_day)
    pm25_all.append(pm25_one_day)

#print(len(pm10_all), len(pm25_all))

# learning
# y = wx + b
# L = (y-y^)**2
b = 0.0
w = np.random.rand(9,)
w *= 0.001
lr=0.01
it_num=1000

loss_iter = []
for it in range(it_num):
    lr_b = 0.0
    lr_w = np.zeros(9,)
    loss = 0
    db = 0.0
    dw = np.zeros(9,)
    for day in range(int(num_day/2)):
        for hour in range(num_hour-9):
            #x1 = pm10_all[day][hour:hour+9]
            x2 = np.array(pm25_all[day][hour:hour+9], dtype=float)
            y = float(pm25_all[day][hour+9])
            #print("x2.shape=", x2.shape)
            #print(np.dot(x2, w))
            diff = np.dot(x2, w)+b-y
            loss += diff**2
            db += 2*diff
            dw += 2*diff*x2
    #print("w.shape=", w.shape)

            lr_b += db**2
            lr_w += dw**2

    b -= lr*db/(int(num_day/2)*(num_hour-9))/np.sqrt(lr_b)
    #w -= lr/(int(num_day/2)*(num_hour-9))*np.divide(dw/np.sqrt(lr_w))
    w -= np.divide(dw, np.sqrt(lr_w))*lr

    loss_iter.append(loss)

#print(loss_iter[-10:-1])
#plt.plot(loss_iter)
#plt.show()

# validation
#for day in range(num_day/2, num_day):


# testing
print('testing...')
test_data = pd.read_csv('test.csv', header=None, encoding='big5')
num_day = test_data.shape[0]/18
num_hour = 9
y_all = []
for day in range(int(num_day)):
    pm25_one_day = []
    for hour in range(num_hour):
#        pm10_one_day.append(data[str(hour)][8+day*18])
        pm25_one_day.append(data[str(hour+2)][9+day*18])
    #print(pm10_one_day)
    #print(pm25_one_day)
    #pm10_all.append(pm10_one_day)
    x = np.array(pm25_one_day, dtype=float)
    y_hat = np.dot(x, w)+b
    y_all.append(y_hat)

#print(len(y_all))

id_array = []
for day in range(int(num_day)):
    id_array.append('id_'+str(day))
ans_np = np.array([id_array, y_all])
ans_dataframe = pd.DataFrame(ans_np.T)
ans_dataframe.to_csv('ans.csv', index=False, header=['id','value'])
