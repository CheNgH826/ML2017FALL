import math
import pandas as pd
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from model import *

RATINGS_CSV_FILE = 'data/train.csv'
MODEL_FILE = 'model_bias'
K_FACTORS = 128
normalize = True
bias = True

ratings = pd.read_csv(RATINGS_CSV_FILE, encoding='utf-8')
max_userid = ratings['UserID'].drop_duplicates().max()
max_movieid = ratings['MovieID'].drop_duplicates().max()
print(len(ratings), 'ratings loaded.')

shuffled_ratings = ratings.sample(frac=1.)
Users = shuffled_ratings['UserID'].values
print('Users:', Users, ', shape =', Users.shape)
Movies = shuffled_ratings['MovieID'].values
print('Movies:', Movies, ', shape =', Movies.shape)
Ratings = shuffled_ratings['Rating'].values
with open('mean_std', 'w') as f:
    f.write(str(np.mean(Ratings))+'\n'+str(np.std(Ratings))+'\n')
    f.write(str(max_userid)+'\n'+str(max_movieid)+'\n')

if normalize:
    Ratings = (Ratings-np.mean(Ratings))/np.std(Ratings)
print('Ratings:', Ratings, ', shape =', Ratings.shape)

model = mf_build(max_userid, max_movieid, K_FACTORS, bias)
#model = DeepModel(max_userid, max_movieid, K_FACTORS)
model.compile(loss='mse', optimizer='adam')

callbacks = [EarlyStopping('val_loss', patience=2), 
             ModelCheckpoint(MODEL_FILE, save_best_only=True)]
history = model.fit([Users, Movies], Ratings, batch_size=512, epochs=30, validation_split=.2, verbose=1, callbacks=callbacks)
pd.DataFrame.from_dict(history.history).to_csv('history.csv', index=False)

min_val_loss, idx = min((val, idx) for (idx, val) in enumerate(history.history['val_loss']))
print('Minimum RMSE at epoch', '{:d}'.format(idx+1), '=', '{:.4f}'.format(math.sqrt(min_val_loss)))
