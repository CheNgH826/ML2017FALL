from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd

clf = joblib.load('para/RandomForestClf.pkl')
x_test = np.array(pd.read_csv('data/proc_x_test'))
ans = clf.predict(x_test)
ans = np.vstack((np.arange(1,len(ans)+1), ans)).T
pd.DataFrame(ans).to_csv('ans/RandomForestClf_ans',index=False,header=['id','label'])