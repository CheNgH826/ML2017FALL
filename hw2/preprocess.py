import numpy as np
import pandas as pd
from util import *

x_data = np.array(pd.read_csv('data/X_train'))
y_data = np.array(pd.read_csv('data/Y_train'))

x_test = np.array(pd.read_csv('data/X_test'))
x_data_and_test = np.concatenate((x_data, x_test), axis=0)
x_analog, x_digital = np.split(x_data_and_test.T,[6])
x_std = np.concatenate((standardize(x_analog.T), x_digital.T), axis=1)
#print(x_std.shape)
#print(x_std)

va_por = 0.33 # validation porprotion

x_train = np.array(x_std[:int(x_data.shape[0]*(1-va_por))])
y_train = np.array(y_data[:int(y_data.shape[0]*(1-va_por))])
x_va = np.array(x_std[int(x_data.shape[0]*(1-va_por)):x_data.shape[0]])
y_va = np.array(y_data[int(y_data.shape[0]*(1-va_por)):])

x_test = x_std[x_data.shape[0]:]

x_train_csv = pd.DataFrame(x_train)
y_train_csv = pd.DataFrame(y_train)
x_va_csv = pd.DataFrame(x_va)
y_va_csv = pd.DataFrame(y_va)
x_test_csv = pd.DataFrame(x_test)

x_train_csv.to_csv('data/proc_x_train', index=False)
y_train_csv.to_csv('data/proc_y_train', index=False)
x_va_csv.to_csv('data/proc_x_va', index=False)
y_va_csv.to_csv('data/proc_y_va', index=False)
x_test_csv.to_csv('data/proc_x_test', index=False)
