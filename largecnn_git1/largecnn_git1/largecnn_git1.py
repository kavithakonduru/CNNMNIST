#Kavitha Konduru project assignment
#This program is for to build and test the nueral network for Handwritten digit classification
# Code for creating a model which can recognize the digits  and uses the MNIST Dataset 

# Import the required libraries
import numpy as np
import numpy
import matplotlib.pyplot as plt
import matplotlib.pyplot as plt
import os
import struct
import gzip
import cv2
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
np.random.seed(seed)
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

#Define the model
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

# Fit the model and save it to history
history=model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)

# Final evaluation of the model
scores = model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

score = model.evaluate(X_test, y_test, verbose=0)
print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])

#save the model
model.save('my_model1.h5')

#del model
#model= load_model ('my_model1.h5')

# Fit the model
#history = model.fit(X, Y, validation_split=0.33, epochs=150, batch_size=10, verbose=0)
# list all data in history
print(history.history.keys())
# summarize history for accuracy
plt.plot(history.history['acc'])
plt.plot(history.history['val_acc'])
plt.title('Model accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('Model loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


#testing with MNIST test data and making predictions
# grab some test images from the test data
#X_test, y_test
test_images = X_test[10:14]

# reshape the test images to standard 28x28 format
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print ('[INFO] test images shape - {}'.format(test_images.shape))

# loop over each of the test images
for i, test_image in enumerate(test_images, start=1):
	# grab a copy of test image for viewing
	org_image = test_image
	
	# reshape the test image to [1x784] format so that our model understands
	test_image = test_image.reshape(1,1,28,28)
    #prediction = model.predict_classes(test_image.reshape(1,1,28,28))
    #prediction = model.predict_classes(test_image.reshape(1,1,28,28))
	
	# make prediction on test image using our trained model
    #prediction = loaded_model.predict_classes(test_image.reshape(1,1,28,28))    
	prediction = model.predict_classes(test_image, verbose=0)	
	# display the prediction and image
    #print('[INFO] Test loss:', score[0])
	print ('[INFO] I think the digit is - {}'.format(prediction[0]))
	plt.subplot(220+i)
	plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()

#recognizing the manually handwritten image or my own image and make predictions
#write the image in paint with pencil and size it to the 28X28 pixels. convert the rgb image to gray image
test = cv2.imread('C:/Users/kavit/Pictures/hrgray2.png',0)
arr = numpy.array(test).reshape(1,1,28,28)
arr = numpy.expand_dims(arr, axis=0)
plt.imshow(test, cmap=plt.get_cmap('gray'))
plt.show()
#test = cv2.cvtColor( test, cv2.COLOR_RGB2GRAY )
test = test.reshape(1, 1, 28, 28)
test = cv2.bitwise_not(test)
pred = loaded_model.predict_classes(test)
print(pred)
print ('[INFO] I think the digit is - {}'.format(pred[0]))

#read one more handwritten image
test1 = cv2.imread('C:/Users/kavit/Pictures/hrgray3.png',0)
arr = numpy.array(test1).reshape(1,1,28,28)
arr = numpy.expand_dims(arr, axis=0)
plt.imshow(test1, cmap=plt.get_cmap('gray'))
plt.show()
print(pred)

test = cv2.imread('C:/Users/kavit/Pictures/hrgray3.png',0)
#test = cv2.cvtColor( test, cv2.COLOR_RGB2GRAY )
test = test.reshape(1, 1, 28, 28)
test = cv2.bitwise_not(test)
pred = model.predict_classes(test, verbose=0)
print(pred)
print ('[INFO] I think the digit is - {}'.format(pred[0]))

test1 = cv2.imread('C:/Users/kavit/Pictures/hrgray3.png',0)
arr = numpy.array(test1).reshape(1,1,28,28)
arr = numpy.expand_dims(arr, axis=0)
plt.imshow(test1, cmap=plt.get_cmap('gray'))
plt.show()
print(pred)

#End of the program