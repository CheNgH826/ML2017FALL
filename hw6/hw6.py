from __future__ import print_function

from keras.layers import Input, Dense
from keras.models import Model, load_model
from keras.optimizers import Adam
from keras.callbacks import Callback, EarlyStopping, ModelCheckpoint
from sklearn.cluster import KMeans

import sys
import os

import pandas as pd
import numpy as np

# build model
input_img = Input(shape=(784,))
encoded = Dense(392, activation='relu')(input_img)
encoded = Dense(128, activation='relu')(encoded)
encoded = Dense(32, activation='relu')(encoded)

decoded = Dense(128, activation='relu')(encoded)
decoded = Dense(392, activation='relu')(decoded)
decoded = Dense(784, activation='linear')(decoded)

# build encoder
encoder = Model(input=input_img, output=encoded)

# build autoencoder
adam = Adam(lr=5e-4)
autoencoder = Model(input=input_img, output=decoded)
autoencoder.compile(optimizer=adam, loss='mse')
autoencoder.summary()

# load images
train_num = 130000
X = np.load(sys.argv[1]) # image.npy
X = X.astype('float32') / 255.
X = np.reshape(X, (len(X), -1))
x_train = X[:train_num]
x_val = X[train_num:]
x_train.shape, x_val.shape

# train autoencoder
MODEL_FILE = 'encoder_model.h5'
if os.path.isfile(MODEL_FILE):
    encoder = load_model(MODEL_FILE)
else:
    callbacks = [EarlyStopping('val_loss', patience=5), 
                 ModelCheckpoint(MODEL_FILE, save_best_only=True)]
    autoencoder.fit(x_train, x_train, epochs=300, batch_size=512, shuffle=True, validation_data=(x_val, x_val), callbacks=callbacks)
    encoder.save(MODEL_FILE)
    

# after training, use encoder to encode image, and feed it into Kmeans
encoded_imgs = encoder.predict(X)
encoded_imgs = encoded_imgs.reshape(encoded_imgs.shape[0], -1)
kmeans = KMeans(n_clusters=2, random_state=0).fit(encoded_imgs)

# get test cases
f = pd.read_csv(sys.argv[2]) # test_case.csv
IDs, idx1, idx2 = np.array(f['ID']), np.array(f['image1_index']), np.array(f['image2_index'])
# predict
o = open(sys.argv[3], 'w') # prediction file path
o.write("ID,Ans\n")
for idx, i1, i2 in zip(IDs, idx1, idx2):
    p1 = kmeans.labels_[i1]
    p2 = kmeans.labels_[i2]
    if p1 == p2:
        pred = 1  # two images in same cluster
    else: 
        pred = 0  # two images not in same cluster
    o.write("{},{}\n".format(idx, pred))
o.close()
