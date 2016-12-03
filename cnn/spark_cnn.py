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
        self.B = B # number of batches
        classifications = load_classifications()
        C = len(classifications)
        self.init_layers(C)
        self.C = C
        # create spark context
        spark = SparkSession.builder.appName('cnn').getOrCreate()
        self.sc = spark.sparkContext

    def train(self, size = 1000):
        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        N = size
        B = self.B
        G = N // B # number of images processed for each batch
        self.G = G

        print('Start training CNN with Spark...')
        print('Training data size: %d' % N)

        time_begin = time()
        sc = self.sc

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


        for i in range(0, self.I):
            print('iteration %d:' % i)
            # forward
            start = time()
            R = sc.parallelize(range(B)).map(forward_map).reduce(forward_reduce)
            R4 = R[0]
            Y = R[1]
            end = time()
            print('forward %.3f' % (end - start))
            return R4, Y
