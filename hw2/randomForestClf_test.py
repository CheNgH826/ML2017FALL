from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import sys
import numpy as np
import pandas as pd

clf = joblib.load('para/RandomForestClf.pkl')
# x_test = np.array(pd.read_csv('data/proc_x_test'))
# x_test = pd.read_csv('data/X_test')
# for i in range(3):
    # print(sys.argv[i])
x_test = pd.read_csv(sys.argv[1])
norm_para = np.array(pd.read_csv('para/norm_para'))
x_analog, x_digital = np.split(x_test.T,[6])
x_analog_norm = ((x_analog.T - norm_para[0])/norm_para[1])
x_test = np.concatenate((x_analog_norm, x_digital.T), axis=1)
ans = clf.predict(x_test)
ans = np.vstack((np.arange(1,len(ans)+1), ans)).T
# pd.DataFrame(ans).to_csv('ans/RandomForestClf_ans',index=False,header=['id','label'])
pd.DataFrame(ans).to_csv(sys.argv[2], index=False,header=['id','label'])
