import numpy as np
import sys
from keras.models import Sequential, load_model
from gensim.models import Word2Vec

#### Parsing data
test_data_file = sys.argv[1]
output_file = sys.argv[2]

test_sen = []
with open(test_data_file, encoding='utf-8') as f:
    for line in list(f):
        test_sen.append(line[line.find(',')+1:-1])
del test_sen[0]

gensim_model = 'w2v.mdl'
w2v_model = Word2Vec.load(gensim_model)

# padding
MAX_SEQUENCE_LENGTH = 30
WORDVEC_DIM = 256
x_test = np.zeros((len(test_sen), MAX_SEQUENCE_LENGTH, WORDVEC_DIM)).astype(float)
empty_word = np.zeros((WORDVEC_DIM)).astype(float)

for i, sen in enumerate(test_sen):
    for j, word in enumerate(sen.split()):
        if j == MAX_SEQUENCE_LENGTH:
            break
        else:
            if word in w2v_model:
                x_test[i,j,:] = w2v_model[word]
            else:
                x_test[i,j,:] = empty_word

trained_model = 'lstm_model.hdf5'
model = load_model(trained_model)
hard_ans = model.predict_classes(x_test)

with open(output_file, 'w') as f:
    f.write('id,label\n')
    for i, ansi in enumerate(hard_ans):
        f.write('{0},{1}\n'.format(i, ansi[0]))
