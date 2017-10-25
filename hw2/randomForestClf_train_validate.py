from sklearn.ensemble import RandomForestClassifier
from sklearn.externals import joblib
import numpy as np
import pandas as pd

x_train = np.array(pd.read_csv('data/proc_x_train'))
y_train = np.array(pd.read_csv('data/proc_y_train')).ravel()

clf = RandomForestClassifier(n_estimators=17, n_jobs=-1, max_depth=18)
clf.fit(x_train, y_train)
print(clf.score(x_train, y_train))
x_va = np.array(pd.read_csv('data/proc_x_va'))
y_va = np.array(pd.read_csv('data/proc_y_va'))
score_his = []
for i in range(20):
    clf.fit(x_train, y_train)
    print(clf.score(x_va, y_va))
    score_his.append(clf.score(x_va, y_va))
print(np.mean(score_his))

# x_test = np.array(pd.read_csv('data/proc_x_test'))
# ans = clf.predict(x_test)
# ans = np.vstack((np.arange(1,len(ans)+1), ans)).T
# joblib.dump(clf, 'para/RandomForestClf.pkl')
# pd.DataFrame(ans).to_csv('ans/RandomForestClf_ans',index=False,header=['id','label'])