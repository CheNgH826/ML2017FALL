import numpy as np
import pandas as pd
from util import *

x_test = pd.read_csv('data/proc_x_test')
x_test = np.concatenate((x_test, np.ones((x_test.shape[0],1))), axis=1)
w = np.array(pd.read_csv('para/generative_para')).T
ans = np.zeros(x_test.shape[0])

for i in range(x_test.shape[0]):
    z0 = np.dot(w[0], x_test[i])
    z1 = np.dot(w[1], x_test[i])
    if sigmoid(z1)>sigmoid(z0):
        ans[i] = 1

ans = ans.reshape(-1, 1).astype(int)
ans = np.concatenate((np.arange(1,len(ans)+1).reshape(ans.shape), ans),axis=1)
pd.DataFrame(ans).to_csv('ans/generatvie_ans',index=False,header=['id','label'])