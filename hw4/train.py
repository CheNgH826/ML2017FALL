import numpy as np
import pandas as pd
import os
import sys
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation
from keras.layers.recurrent import LSTM
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LambdaCallback
from gensim.models import Word2Vec

# I/O path
train_file = sys.argv[1]
train_unlabel_file = sys.argv[2]
outdir = 'output'

#### Parsing data
label = []
train_sen = []
with open(train_file, 'r', encoding='utf-8') as f:
    for line in list(f):
        label.append(line[0])
        train_sen.append(line[10:-1])

unlabel_sen = []
with open(train_unlabel_file, 'r', encoding='utf-8') as f:
    unlabel_sen = list(f)

#### Word and sequence embedding
MAX_SEQUENCE_LENGTH = 30
VALIDATION_SPLIT = 0.2
WORDVEC_DIM = 256

texts = []
for sen in train_sen + unlabel_sen:
    texts.append(sen.split())

gensim_model = 'w2v.mdl'
if os.path.isfile(gensim_model):
    print('load w2v...')
    w2v_model = Word2Vec.load(gensim_model)
else:
    print('train new w2v...')
    w2v_model = Word2Vec(texts, min_count=16, size=WORDVEC_DIM, iter=20)
    w2v_model.init_sims(replace=True)
    w2v_model.save(gensim_model)
    print('w2v ready!')

# padding
x_data = np.zeros((len(train_sen), MAX_SEQUENCE_LENGTH, WORDVEC_DIM)).astype(float)
empty_word = np.zeros((WORDVEC_DIM)).astype(float)

for i, sen in enumerate(train_sen):
    for j, word in enumerate(sen.split()):
        if j == MAX_SEQUENCE_LENGTH:
            break
        else:
            if word in w2v_model:
                x_data[i,j,:] = w2v_model[word]
            else:
                x_data[i,j,:] = empty_word

# split the data into a training set and a validation set
indices = np.arange(len(train_sen))
np.random.shuffle(indices)
x_data = x_data[indices]
label = np.array(label)[indices]
nb_validation_samples = int(VALIDATION_SPLIT * len(train_sen))

x_train = x_data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_val = x_data[-nb_validation_samples:]
y_val = label[-nb_validation_samples:]

### Model output preparation
if not os.path.exists(outdir):
    os.makedirs(outdir)
model_path = outdir + '/model-{epoch:02d}-{val_acc:.3f}.hdf5'
checkpoint = ModelCheckpoint(model_path, monitor='val_acc', verbose=1, save_best_only=True, mode='max')
earlystop = EarlyStopping(monitor='val_loss', patience=2, verbose=1)
batch_print_callback = LambdaCallback(
    on_epoch_end=lambda batch, logs: print(
        '\nINFO:root:Epoch[%d] Train-accuracy=%f\nINFO:root:Epoch[%d] Validation-accuracy=%f' %
        (batch, logs['acc'], batch, logs['val_acc'])))

callbacks_list = [checkpoint, batch_print_callback, earlystop]

#### Model building
trained_model = 'lstm_model.hdf5'
if os.path.isfile(trained_model):
    print('load existing model...')
    model = load_model(trained_model)
else:
    print('train new...')
    model = Sequential()
    model.add(LSTM(256, input_shape=(MAX_SEQUENCE_LENGTH, WORDVEC_DIM), dropout=0.3, recurrent_dropout=0., return_sequences=True))
    model.add(LSTM(256, dropout=0.3, recurrent_dropout=0.))
    model.add(Dense(1, activation='sigmoid'))
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    model.summary()
    history = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=30, batch_size=128, callbacks=callbacks_list)
    w = pd.DataFrame.from_dict(history.history)
    w.to_csv(outdir + '/history.csv')
