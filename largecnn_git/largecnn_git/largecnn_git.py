from __future__ import print_function
from __future__ import division

import keras
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras import backend as K

batch_size = 128
num_classes = 10 #how many image classes defined in the program 
epochs = 20

#input image dimenssion
img_rows, img_cols = 28,28

# shuffle and split mnist between and to train and test sets
(X_train, Y_train), (X_test, Y_test) = mnist.load_data()

if K.image_data_format()== 'channels_first':
	X_train = X_train.reshape(X_train.shape[0],1, img_rows,img_cols)
	X_test = X_test.reshape(X_test.shape[0],1, img_rows, img_cols)
	input_shape= (1,img_rows, img_cols)
else :
	X_train= X_train.reshape(X_train.shape[0],img_rows, img_cols,1)
	X_test= X_test.reshape(X_test.shape[0],img_rows, img_cols,1)
	input_shape = (img_rows, img_cols,1)

X_train = X_train.astype('float32') #declaring the data type of 
X_test = X_test.astype('float32')
X_train /= 255
X_test /= 255
print(X_train.shape[0], 'train samples')
print(X_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
Y_train = keras.utils.to_categorical(Y_train, num_classes)
Y_test = keras.utils.to_categorical(Y_test, num_classes)

model=Sequential()
model.add(Conv2D(32, kernel_size=(3,3), activation= 'relu',input_shape=input_shape))
model.add(Conv2D(64, (3,3), activation= 'relu'))
model.add(MaxPooling2D(pool_size=(2,2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

model.compile(loss=keras.losses.categorical_crossentropy, optimizer=keras.optimizers.Adadelta(), metrics=['accuracy'])

model.fit(X_train, Y_train,batch_size=batch_size, epochs=epochs, verbose=1, validation_data=(X_test, Y_test))

from keras.models import load_model

model.save('my_model1.h5')
del model

model= load_model ('my_model1.h5')

score = model.evaluate(X_test, Y_test, verbose=0)

print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])