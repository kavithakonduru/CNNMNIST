
# Larger CNN for the MNIST Dataset
import numpy
import os;
import cv2;

from keras.models import model_from_json
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils
from keras import backend as K

K.set_image_dim_ordering('th')
# fix random seed for reproducibility
seed = 7
numpy.random.seed(seed)
# load data
(X_train, y_train), (X_test, y_test) = mnist.load_data()
# reshape to be [samples][pixels][width][height]
X_train = X_train.reshape(X_train.shape[0], 1, 28, 28).astype('float32')
X_test = X_test.reshape(X_test.shape[0], 1, 28, 28).astype('float32')
# normalize inputs from 0-255 to 0-1
X_train = X_train / 255
X_test = X_test / 255
# one hot encode outputs
y_train = np_utils.to_categorical(y_train)
y_test = np_utils.to_categorical(y_test)
num_classes = y_test.shape[1]
def larger_model():
	# create model
	model = Sequential()
	model.add(Conv2D(30, (5, 5), input_shape=(1, 28, 28), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Conv2D(15, (3, 3), activation='relu'))
	model.add(MaxPooling2D(pool_size=(2, 2)))
	model.add(Dropout(0.2))
	model.add(Flatten())
	model.add(Dense(128, activation='relu'))
	model.add(Dense(50, activation='relu'))
	model.add(Dense(num_classes, activation='softmax'))
	# Compile model
	model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
	return model
# build the model
model = larger_model()
# Fit the model
model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))
from keras.models import load_model

model.save('my_model1.h5')
#del model

# model= load_model ('my_model1.h5')

score = model.evaluate(X_test, y_test, verbose=0)

print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])
import matplotlib.pyplot as plt

#for i in range(0, 9): 
 #   plt.subplot(330 + 1 + i) 
  #  plt.imshow(X_train[i], cmap=plt.get_cmap('gray'))
# show the plot
#plt.show()
#import matplotlib.pyplot as plt
#for i in range(0, 9): 
 #   plt.subplot(330 + 1 + i) 
  #  plt.imshow(X_test[i], cmap=plt.get_cmap('gray'))
# show the plot
#plt.show()

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")
 
# later...
 
# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))

im = cv2.imread('C:\\Users\\kavit\\Pictures\\zerokk0.png',0)
im = im/255
pr = model.predict_classes(im.reshape((1, 1, 28, 28)))
#pr = loaded_model.predict_classes(im)
print(pr)

img = cv2.imread('C:\\Users\\kavit\\Pictures\\zerokk0.png',0)
cv2.namedWindow('image', cv2.WINDOW_NORMAL)
cv2.imshow('image',img)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('messigray.png',img)

cv2.imshow('image',pr)
cv2.waitKey(0)
cv2.destroyAllWindows()
cv2.imwrite('zerogray.png',);
