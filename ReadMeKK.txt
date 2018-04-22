Kavitha Konduru: Project assignment
Description: Build a Neural network model for Handwritten digit classification using MNIST data set and the Python frame work Keras and Tensorflow
Course: Deep Learning
Instructor: Dr Larry Pyeatt
**********************************************************************************

1. The project folder with the code files and the paper are placed at the following github location:
https://github.com/kavithakonduru/CNNMNIST

2. The CNN model is written in python using the frame work Keras, Tensorflow and also uses the numpy. The total project is written and executed from Visual studio 
environment which works well on 'CPU'. Assuming the following libraries or Python with the Keras, Tensorflow and other libraries are already installed.
numpy, matplotlib.pyplot,os, gzip, cv2, and the following:
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv2D
from keras.layers.convolutional import MaxPooling2D
from keras.utils import np_utils


3. Instructions to run the code files: navigate to the folder https://github.com/kavithakonduru/CNNMNIST/tree/master/HRDigitclassification_KavithaKonduru/HRDigitclassification_KavithaKonduru
 contains the project folders for 

a) The CNN with ReLu and SoftMax activation funnctions : HRDigitclassification_KavithaKonduru->HRDigitclassification_KavithaKonduru.py
	Please copy the total folder and run the python file because the model is using the my handwritten images  which are stored and should be at the same level of the .py files.

b) The predictors for the already loaded model: HRDigitpredictors_KavithaKonduru-> HRDigitclassification_KavithaKonduru.pyproj

c) The CNN with ReLu and Sigmoid activation funnctions: https://github.com/kavithakonduru/CNNMNIST/tree/master/HRDR_CNN_sigmoid_KK

4. The model used the MNIST data and the following files should be loaded to the system as a prerequisite of the network to be run. 
   a) http://yann.lecun.com/exdb/mnist/
   b) Four files are available on this site: 
		train-images-idx3-ubyte.gz:  training set images (9912422 bytes) 
		train-labels-idx1-ubyte.gz:  training set labels (28881 bytes) 
		t10k-images-idx3-ubyte.gz:   test set images (1648877 bytes) 
		t10k-labels-idx1-ubyte.gz:   test set labels (4542 bytes)
		
5. The results document with the screenshots is also stored in the github folder

6. The Paper written about the 'Handwritten digit Recognition using CNN' is stored and can be accessed from the github folder https://github.com/kavithakonduru/CNNMNIST/paper