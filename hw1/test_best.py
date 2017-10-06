import sys
import pandas as pd
import numpy as np

#para_csv = pd.read_csv('paras_for_6.78.csv')
if len(sys.argv) != 3:
    print('invalid argument numbers')
    exit()

para = np.genfromtxt('paras_best.csv', delimiter=',')

w = np.array(para[:-1])
b = para[-1]

test_data = pd.read_csv(sys.argv[1], header=None, encoding='big5')
#test_data = pd.read_csv('test.csv', header=None, encoding='big5')
num_day = test_data.shape[0]/18
num_hour = 9
y_ans=[]
for day in range(int(num_day)):
    x = []
    for h in range(num_hour):
        if test_data[h+2][10+day*18] == 'NR':
            test_data[h+2][10+day*18] = '0'

    xi = []
    for hour in range(num_hour):
        xi.append(test_data[hour+2][day*18+9])
    x.append(xi)
    x = np.array(x, dtype=float)
    x = np.concatenate((x, np.square(x[:, -5:])), axis=1)
    #x = np.concatenate((x, np.square(x)), axis=1)
    y_hat = np.dot(x, w)+b
    y_ans.append(y_hat[0])

id_array = []
for day in range(int(num_day)):
    id_array.append('id_'+str(day))
ans_np = np.array([id_array, y_ans])
ans_dataframe = pd.DataFrame(ans_np.T)
#ans_dataframe.to_csv('res.csv', index=False, header=['id','value'])
ans_dataframe.to_csv(sys.argv[2], index=False, header=['id','value'])
