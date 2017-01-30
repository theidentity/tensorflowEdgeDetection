from __future__ import division, print_function, absolute_import

import tflearn
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.normalization import local_response_normalization
from tflearn.layers.estimator import regression
import numpy as np

# Data loading and preprocessing
trainSet = np.load("train.npy")
trainLabel = np.load("label.npy")
testSet = np.load("train.npy")
testLabel = np.load("label.npy")

X, Y, testX, testY = trainSet,trainLabel,testSet,testLabel
X = X.reshape([-1, 32, 32, 1])
testX = testX.reshape([-1, 32, 32, 1])
batchsize = 100
Y = Y.reshape([-1,batchsize,1])
testY = testY.reshape([-1,batchsize,1])
print(testX.shape)
print(testY.shape)

# Building convolutional network
network = input_data(shape=[None, 32, 32, 1], name='input')
network = conv_2d(network, 32, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = conv_2d(network, 64, 3, activation='relu', regularizer="L2")
network = max_pool_2d(network, 2)
network = local_response_normalization(network)
network = fully_connected(network, 128, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 256, activation='tanh')
network = dropout(network, 0.8)
network = fully_connected(network, 1, activation='linear')
network = regression(network, optimizer='adam', learning_rate=0.01,
                 loss='mean_square', name='target')

# Training
model = tflearn.DNN(network, tensorboard_verbose=2)
model.fit({'input': X}, {'target': Y}, n_epoch=20,
           validation_set=({'input': testX}, {'target': testY}),
           snapshot_step=100, show_metric=True, run_id='convnet_mnist')

model.save("model1.tfl")
# model.load("model1.tfl")