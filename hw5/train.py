import math
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
#from CFModel import CFModel
from model import *

RATINGS_CSV_FILE = 'data/train.csv'
MODEL_WEIGHTS_FILE = 'model/{val_loss:.3f}'
K_FACTORS = 256
normalize = False
bias = False

ratings = pd.read_csv(RATINGS_CSV_FILE, encoding='utf-8')
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print(len(ratings), 'ratings loaded.')

shuffled_ratings = ratings.sample(frac=1.)#, random_state=RNG_SEED)
Users = shuffled_ratings['UserID'].values
#Users = ratings['UserID']
print('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['MovieID'].values
#Movies = ratings['MovieID']
print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['Rating'].values
#Ratings = ratings['Rating']
with open('mean_std', 'w') as f:
    f.write(str(np.mean(Ratings))+'\n'+str(np.std(Ratings))+'\n')
    f.write(str(max_userid)+'\n'+str(max_movieid)+'\n')

if normalize:
    Ratings = (Ratings-np.mean(Ratings))/np.std(Ratings)
print('Ratings:', Ratings, ', shape =', Ratings.shape)


#model = CFModel(max_userid, max_movieid, K_FACTORS)
#model = DeepModel(max_userid, max_movieid, K_FACTORS)
model = mf_build(max_userid, max_movieid, K_FACTORS, bias)
model.compile(loss='mse', optimizer='adam')

callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_WEIGHTS_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, batch_size=512, epochs=30, validation_split=.2, verbose=1, callbacks=callbacks)
pd.DataFrame.from_dict(history.history).to_csv('history')

min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))

