import sys
import pandas as pd
import numpy as np
import keras
import h5py
from keras.models import Sequential
from keras.layers.core import Dense , Dropout, Activation
from keras.layers.normalization import BatchNormalization
from keras.layers import AveragePooling2D,Conv2D,ZeroPadding2D, MaxPooling2D ,Flatten
from keras.optimizers import SGD, Adam
from keras.utils import np_utils
from keras import backend
backend.set_image_dim_ordering('th')
#categorical_crossentropy
#start = timeit.default_timer()
from keras.preprocessing.image import ImageDataGenerator

datagen_train = ImageDataGenerator(featurewise_center = True,featurewise_std_normalization = True,rotation_range = 22.0,width_shift_range = 0.1,height_shift_range = 0.1)

datagen_test = ImageDataGenerator(featurewise_center = True,featurewise_std_normalization = True)

def readfile():
        raw = pd.read_csv(sys.argv[1])
        x_train = raw['feature'].as_matrix()
        x_train = np.array([[int(i) for i in x.split()] for x in x_train])
        x_train = x_train.astype('float32')
        x_train /= 255
        x_train = x_train.reshape(x_train.shape[0],1,48,48)

        y_train = raw['label'].as_matrix()
        y_train = keras.utils.to_categorical(y_train, 7)

        return x_train, y_train

x_train , y_train = readfile()
# datagen preprocess
datagen_train.fit(x_train)

# optimizer
adam = keras.optimizers.Adam(lr = 0.001, beta_1 = 0.9 ,beta_2 = 0.999,epsilon = 1e-08 ,decay = 0.0)
adamax = keras.optimizers.Adamax(lr = 0.002 ,beta_1 = 0.9 , beta_2 = 0.999 , epsilon = 1e-08 ,decay = 0.0)
adadelta = keras.optimizers.Adadelta(lr =0.001 ,rho = 0.95 ,epsilon = 1e-08)

# model building

model = Sequential()
#for i in range(3):
model.add(Conv2D(64,5,5,activation = "relu",input_shape = (1,48,48)))
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding = (2,2),data_format = "channels_first"))
model.add(MaxPooling2D((3,3),strides=(2,2)))
model.add(ZeroPadding2D(padding = (1,1),data_format = "channels_first"))

model.add(Conv2D(64,3,3,activation = "relu"))
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding = (1,1),data_format = "channels_first"))

model.add(Conv2D(64,3,3,activation = "relu"))
model.add(BatchNormalization())
model.add(AveragePooling2D(pool_size=(2,2)))
model.add(ZeroPadding2D(padding = (1,1),data_format = "channels_first"))

model.add(Conv2D(128,2,2,activation = "relu"))
model.add(BatchNormalization())
model.add(ZeroPadding2D(padding = (1,1),data_format = "channels_first"))

model.add(Conv2D(128,3,3,activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3),strides=(2,2)))
model.add(ZeroPadding2D(padding = (2,2),data_format = "channels_first"))
model.add(AveragePooling2D(pool_size=(1,1)))

model.add(Conv2D(256,3,3,activation = "relu"))
model.add(BatchNormalization())
model.add(MaxPooling2D((3,3)))
model.add(ZeroPadding2D(padding = (1,1),data_format = "channels_first"))

model.add(Flatten())

for i in range(1):
	model.add(Dense(units = 2048,activation = 'relu'))
	model.add(BatchNormalization())	
	model.add(Dropout(0.4))
	model.add(Dense(units = 1024,activation = 'relu'))
	model.add(BatchNormalization())
	model.add(Dropout(0.3))

model.add(Dense(units = 7,activation = 'softmax'))
model.compile(loss = 'categorical_crossentropy', optimizer = adamax,metrics = ['accuracy'])

model.summary()

model.fit_generator(datagen_train.flow(x_train,y_train,batch_size = 128,shuffle =True),steps_per_epoch = len(x_train)/128,epochs = 70,validation_steps = 0.12,verbose = 1)
model.save('model.hdf5')
