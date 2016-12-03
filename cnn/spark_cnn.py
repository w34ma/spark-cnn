# spark enabled CNN
from pyspark.sql import SparkSession

import numpy as np
from time import time
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *

from cnn import CNN

class SparkCNN(CNN):
    def __init__(self, I, B):
        CNN.__init__(self, I)
        self.name = 'spark_cnn'
        self.B = B # number of batches
        # create spark context
        spark = SparkSession.builder.appName('cnn').getOrCreate()
        self.sc = spark.sparkContext

    def forward(self, X):
        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc

        N, W, H, D = X.shape
        G = N // self.B # number of images in each batch

        # define forward funcion for spark map
        def forward_map(batch):
            start = batch * G
            end = start + G
            X, Y = load_training_data(start, end)
            R1 = conv.forward(X)
            # save X
            save(X, 'X_batch_' + str(batch))
            X = None
            R2 = relu.forward(R1)
            # save R1
            save(R1, 'R1_batch_' + str(batch))
            R1 = None
            R3 = pool.forward(R2)
            # save R2
            save(R2, 'R2_batch_' + str(batch))
            R2 = None
            R4 = fc.forward(R3)
            # save R3
            save(R3, 'R3_batch_' + str(batch))
            R3 = None
            return [R4, Y]

        def forward_reduce(a, b):
            R4 = np.append(a[0], b[0], 0)
            Y = np.append(a[1], b[1], 0)
            return [R4, Y]

        def backward_map(batch):


        def backward_reduce(a, b):


        for i in range(0, self.I):
            print('iteration %d:' % i)
            # forward
            start = time()
            R = sc.parallelize(range(B)).map(forward_map).reduce(forward_reduce)
            R4 = R[0]
            Y = R[1]
            end = time()
            print('forward %.3f' % (end - start))


            # backward
            # calculate loss and gradients
            L, dS = softmax(R4, Y)
            start = time()
            batches = []
            for i in range(0, B):
                dS_i = dS[i * G:i * G + G, :]
                batches.append(dS_i)

            R = sc.parallelize(batches).map(backward_map).reduce(backward_reduce)

            dAConv = R[0]
            dbConv = R[1]
            dAFC = R[2]
            dbFC = R[3]
            end = time()
            print('backward %.3f' % (end - start))

            # update parameters
            # velocities
            conv.V = self.mu * conv.V - self.rho * dAConv
            fc.V = self.mu * fc.V - self.rho * dAFC

            # weights
            conv.A += conv.V
            fc.A += fc.V

            # biases
            conv.b += (0 - self.rho) * dbConv
            fc.b += (0 - self.rho) * dbFC

            # save parameters
            self.conv = conv
            self.relu = relu
            self.pool = pool
            self.fc = fc
            self.save()
