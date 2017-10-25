import pandas as pd
import numpy as np
import sys
from util import *

# x_test = pd.read_csv('data/proc_x_test')
# x_test = pd.read_csv('data/X_test')
x_test = pd.read_csv(sys.argv[1])
norm_para = np.array(pd.read_csv('para/norm_para'))
x_analog, x_digital = np.split(x_test.T,[6])
x_analog_norm = ((x_analog.T - norm_para[0])/norm_para[1])
x_test = np.concatenate((x_analog_norm, x_digital.T), axis=1)
x_test = np.concatenate((x_test, np.ones((x_test.shape[0],1))), axis=1)
w = pd.read_csv('para/logistic_para')

f = sigmoid(np.dot(x_test, w))
ans = np.array(determine_ans(f))
ans = np.concatenate((np.arange(1,len(ans)+1).reshape(ans.shape), ans),axis=1)
# pd.DataFrame(ans).to_csv('ans/logsitic_ans',index=False,header=['id','label'])
pd.DataFrame(ans).to_csv(sys.argv[2],index=False,header=['id','label'])