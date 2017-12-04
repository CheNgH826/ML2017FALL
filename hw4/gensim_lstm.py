import numpy as np
import os
import sys
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Activation, Flatten
from keras.layers.embeddings import Embedding
from keras.layers.recurrent import LSTM
from keras.preprocessing import sequence
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import LambdaCallback

# I/O path
#data_path = os.environ.get('GRAPE_DATASET_DIR')
data_path = os.environ.get('HOME')+'/ML2017FALL/hw4'
outdir = 'output'

#### Parsing data
label = []
train_sen = []
with open(data_path+'/data/training_label.txt', 'r', encoding='utf-8') as f:
    for line in list(f):
        # words = line.split()
        # train_sen.append(' '.join(words[2:]))
        label.append(line[0])
        train_sen.append(line[10:-1])

test_sen = []
with open(data_path+'/data/testing_data.txt', encoding='utf-8') as f:
    for line in list(f):
        test_sen.append(line[line.find(',')+1:-1])
del test_sen[0]

#### Word and sequence embedding
MAX_NB_WORDS = 10000
MAX_SEQUENCE_LENGTH = 25
VALIDATION_SPLIT = 0.2

with open(data_path+'/data/training_nolabel.txt', 'r', encoding='utf-8') as unlabel_sen_f:
    unlabel_sen = list(unlabel_sen_f)
# texts = train_sen + unlabel_sen
texts = train_sen

tokenizer = Tokenizer(num_words=MAX_NB_WORDS)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(train_sen)
test_seq = tokenizer.texts_to_sequences(test_sen)
unlabel_seq = tokenizer.texts_to_sequences(unlabel_sen)

data = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH)
test = pad_sequences(test_seq, maxlen=MAX_SEQUENCE_LENGTH)
unlabel = pad_sequences(unlabel_seq, maxlen=MAX_SEQUENCE_LENGTH)

# label = to_categorical(np.asarray(label))
# print(label[:5])
# print('Shape of data tensor:', data.shape)
# print('Shape of label tensor:', label.shape)

# split the data into a training set and a validation set
indices = np.arange(data.shape[0])
np.random.shuffle(indices)
data = data[indices]
label = np.array(label)[indices]
nb_validation_samples = int(VALIDATION_SPLIT * data.shape[0])

x_train = data[:-nb_validation_samples]
y_train = label[:-nb_validation_samples]
x_val = data[-nb_validation_samples:]
y_val = label[-nb_validation_samples:]

### Model output
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
trained_model = outdir+'/temp.hdf5'
import pandas as pd
if os.path.isfile(trained_model):
    print('load existing model...')
    model = load_model(trained_model)
else:
    print('train new...')
    model = Sequential()
    model.add(Embedding(MAX_NB_WORDS, 128, input_length=MAX_SEQUENCE_LENGTH))
    model.add(LSTM(128, dropout=0.5, recurrent_dropout=0.5))
    model.add(Dense(1, activation='sigmoid'))
    # model.compile(loss='categorical_crossentropy', optimizer='rmsprop', metrics=['acc'])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['acc'])
    hitory = model.fit(x_train, y_train, validation_data=(x_val, y_val), epochs=int(sys.argv[1]), batch_size=64, callbacks=callbacks_list)
    w = pd.DataFrame.from_dict(history.history)
    w.to_csv(outdir + '/history.csv')
model.summary()

### Generate pseudo data
pseudo_ans = model.predict(unlabel)
pseudo_label = []
pseudo_data = []
for i, ansi in enumerate(pseudo_ans):
    if ansi > 0.7:
        pseudo_label.append(1)
        pseudo_data.append(unlabel[i])
    elif ansi < 0.3:
        pseudo_label.append(0)
        pseudo_data.append(unlabel[i])

### Retrain
x_mix = np.concatenate((x_train, pseudo_data), axis=0)
y_mix = np.concatenate((y_train, pseudo_label), axis=0)

history = model.fit(x_mix, y_mix, validation_data=(x_val, y_val), epochs=int(sys.argv[1]), batch_size=64, callbacks=callbacks_list)
ans = model.predict(test)
hard_ans = model.predict_classes(test)
with open(outdir + '/ans.csv', 'w') as f:
    f.write('id,label\n')
    for i, ansi in enumerate(ans):
        f.write('{0},{1}\n'.format(i, ansi[0]))
with open(outdir + '/hard_ans.csv', 'w') as f:
    f.write('id,label\n')
    for i, ansi in enumerate(hard_ans):
        f.write('{0},{1}\n'.format(i, ansi[0]))

w = pd.DataFrame.from_dict(history.history)
w.to_csv(outdir + '/history_semi.csv')
