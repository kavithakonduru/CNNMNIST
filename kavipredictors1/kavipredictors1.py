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

