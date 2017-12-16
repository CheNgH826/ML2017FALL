from model import *
from keras.models import load_model
import pandas as pd
import numpy as np
import sys

RATINGS_CSV_FILE = 'data/train.csv'
TESTING_FILE = 'data/test.csv'
MODEL_FILE = sys.argv[1]
ANS_FILE = sys.argv[2]
K_FACTORS = 90

ratings = pd.read_csv(RATINGS_CSV_FILE, encoding='utf-8')
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
Ratings = ratings['Rating']

testing = pd.read_csv(TESTING_FILE)
usersTest = testing['UserID'].values
moviesTest = testing['MovieID'].values

#trained_model = CFModel(max_userid, max_movieid, K_FACTORS)
#trained_model = DeepModel(max_userid, max_movieid, K_FACTORS)
trained_model = load_model(MODEL_FILE)

ans = trained_model.predict([usersTest, moviesTest])
ans = ans*np.std(Ratings)+np.mean(Ratings)

with open(ANS_FILE, 'w') as f:
    f.write('TestDataID,Rating\n')
    for i, user in enumerate(usersTest):
        f.write(str(i+1)+','+str(ans[i][0])+'\n')
