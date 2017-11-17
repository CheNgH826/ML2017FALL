import keras
import sys
import numpy as np
import pandas as pd
from keras import backend as K

img_rows, img_cols = 48, 48
'''
if K.image_data_format() == 'channels_first':
    input_shape = (1, img_rows, img_cols)
else:
    input_shape = (img_rows, img_cols, 1)
'''

raw = pd.read_csv(sys.argv[1])
test = raw['feature'].as_matrix()
test = np.array([[int(i) for i in x.split()] for x in test])
test = test.reshape(test.shape[0],48,48,1)
test = test.astype('float32')/255

model = keras.models.load_model('model.hdf5')
#model.summary()
ans = model.predict_classes(test, verbose=1)
ans = np.vstack((np.arange(ans.shape[0]), ans))
pd.DataFrame(ans.T).to_csv(sys.argv[2], header=['id', 'label'], index=False)
