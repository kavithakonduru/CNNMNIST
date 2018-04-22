
# Loading the saved model and predicting the results with new test images
#import the all required libraries
import numpy as np
import glob
import numpy
import matplotlib.pyplot as plt
import os
import struct
import gzip
import cv2
from keras.models import model_from_json
from keras.models import load_model
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
# load json and create model
# C:\Users\kavit\source\repos\largecnn_git\kavipredictors\kavipredictors\model.json
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")
 
# evaluate loaded model on test data
loaded_model.compile(loss='binary_crossentropy', optimizer='rmsprop', metrics=['accuracy'])
#history=loaded_model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=10, batch_size=200)
score = loaded_model.evaluate(X_test, y_test, verbose=0)
print("Large CNN Error: %.2f%%" % (100-score[1]*100))
print("%s: %.2f%%" % (loaded_model.metrics_names[1], score[1]*100))


print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])
# build the model

#scores = model.evaluate(X_test, y_test, verbose=0)
#print("Large CNN Error: %.2f%%" % (100-scores[1]*100))

#model.save('my_model1.h5')
#del model

#model= load_model ('my_model1.h5')

#score = model.evaluate(X_test, y_test, verbose=0)

print('[INFO] Test loss:', score[0])
print('[INFO] Test accuracy:', score[1])

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

	# make prediction on test image using our trained model
	prediction = loaded_model.predict_classes(test_image.reshape(1,1,28,28))
	
	# display the prediction and image
   	print ('[INFO] I think the digit is - {}'.format(prediction[0]))
	plt.subplot(220+i)
	plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()


print("helloooooo")
import cv2

#test = cv2.imread('C:\\Users\\kavit\\Pictures\\hrgray3.png')
test = cv2.imread('hrgray2.png',0)
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

#test1 = cv2.imread('C:/Users/kavit/Pictures/hrgray2.png',0)
test1 = cv2.imread('hrgray2.png',0)
arr = numpy.array(test1).reshape(1,1,28,28)
arr = numpy.expand_dims(arr, axis=0)
plt.imshow(test1, cmap=plt.get_cmap('gray'))
plt.show()
#print(pred)
#print ('[INFO] I think the digit is - {}'.format(pred[0]))

print("Now we will test with the image 3 digit")

test = cv2.imread('hrgray3.png',0)
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
test1 = cv2.imread('hrgray3.png',0)
arr = numpy.array(test1).reshape(1,1,28,28)
arr = numpy.expand_dims(arr, axis=0)
plt.imshow(test1, cmap=plt.get_cmap('gray'))
plt.show()
#print(pred)



