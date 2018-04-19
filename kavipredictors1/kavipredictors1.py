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

# load json and create model
# C:\Users\kavit\source\repos\largecnn_git\kavipredictors\kavipredictors\model.json
json_file = open('C:\\Users\\kavit\\source\\repos\\largecnn_git\\kavipredictors\\kavipredictors\\model.json', 'r')
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
import matplotlib.pyplot as plt

# grab some test images from the test data
#X_test, y_test
test_images = X_test[1:5]

# reshape the test images to standard 28x28 format
test_images = test_images.reshape(test_images.shape[0], 28, 28)
print ('[INFO] test images shape - {}'.format(test_images.shape))

# loop over each of the test images
for i, test_image in enumerate(test_images, start=1):
	# grab a copy of test image for viewing
	org_image = test_image
	
	# reshape the test image to [1x784] format so that our model understands
	test_image = test_image.reshape(1,784)
	
	# make prediction on test image using our trained model
	prediction = model.predict_classes(test_image, verbose=0)
	
	# display the prediction and image
    #print('[INFO] Test loss:', score[0])
	print ('[INFO] I think the digit is - {}'.format(prediction[0]))
	plt.subplot(220+i)
	plt.imshow(org_image, cmap=plt.get_cmap('gray'))

plt.show()
