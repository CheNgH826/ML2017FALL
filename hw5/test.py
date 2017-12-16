from model import *
from keras.models import load_model
import pandas as pd
import numpy as np
import sys

TESTING_FILE = sys.argv[1]
MODEL_FILE = sys.argv[2]
ANS_FILE = sys.argv[3]
normalize = True

with open('mean_std', 'r') as f:
    mean = float(f.readline())
    std = float(f.readline())
    max_userid = int(f.readline())
    max_movieid = int(f.readline())

#ratings = pd.read_csv(RATINGS_CSV_FILE, encoding='utf-8')
#max_userid = ratings['UserID'].drop_duplicates().max()
#max_movieid = ratings['MovieID'].drop_duplicates().max()
#Ratings = ratings['Rating']

testing = pd.read_csv(TESTING_FILE)
usersTest = testing['UserID'].values
moviesTest = testing['MovieID'].values

trained_model = load_model(MODEL_FILE)

ans = trained_model.predict([usersTest, moviesTest])
if normalize:
    ans = ans*std+mean

with open(ANS_FILE, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, user in enumerate(usersTest):
        f.write(str(i+1)+','+str(ans[i][0])+'\n')
