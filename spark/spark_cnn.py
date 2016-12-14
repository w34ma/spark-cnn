# spark enabled CNN
from pyspark.sql import SparkSession
import os
import numpy as np
from time import time
from spark.conv import ConvolutionLayer
from spark.relu import ReLULayer
from spark.pool import PoolingLayer
from spark.fc import FCLayer
from spark.utils import *
from spark.cnn import CNN

class SparkCNN(CNN):
    def __init__(self, I, B):
        CNN.__init__(self, I)
        self.name = 'spark_cnn'
        self.B = B # number of batches
        # create spark context
        spark = SparkSession.builder.appName('spark-cnn').getOrCreate()
        self.sc = spark.sparkContext

    def predict_train(self, X):
        self.reload()
        N = X.shape[0]
        B = self.B
        G = N // B

        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc

        XB = sc.broadcast(X)

        def forward_map(batch):
            start = batch * G
            end = start + G

            X = XB.value[start:end, :, :, :]

            R1 = conv.forward(X)
            X = None
            R2 = relu.forward(R1)
            R1 = None
            R3 = pool.forward(R2)
            R2 = None
            R4 = fc.forward(R3)
            R3 = None
            return R4

        def forward_reduce(a, b):
            R4 = np.append(a, b, 0)
            return R4

        R4 = sc.parallelize(range(B), B).map(forward_map).reduce(forward_reduce)
        return R4

    def predict(self, X):
        self.reload()
        N = X.shape[0]
        B = self.B
        G = N // B

        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc

        X, Y = load_testing_data(0, size)
        XB = sc.broadcast(X)

        def forward_map(batch):
            start = batch * G
            end = start + G

            X = XB.value[start:end, :, :, :]

            R1 = conv.forward(X)
            X = None
            R2 = relu.forward(R1)
            R1 = None
            R3 = pool.forward(R2)
            R2 = None
            R4 = fc.forward(R3)
            R3 = None
            return R4

        def forward_reduce(a, b):
            R4 = np.append(a, b, 0)
            return R4

        R4 = sc.parallelize(range(B), B).map(forward_map).reduce(forward_reduce)
        return R4

    def train(self, size = 1000):
        print('Start training CNN with Spark...')
        print('Training data size: %d' % size)

        time_begin = time()
        X, Y = load_training_data(0, size)
        sc = self.sc
        self.XB = sc.broadcast(X)

        time_middle = time()
        print('X broadcasting done, time %.4f' % (time_middle - time_begin))

        for i in range(0, self.I):
            print('iteration %d' % i)

            # forward
            start = time()
            batches, R4 = self.forward(size)
            middle = time()

            # calculate loss and gradients
            L, dS = softmax(R4, Y)

            # backward
            dAConv, dbConv, dAFC, dbFC = self.backward(batches, dS)
            end = time()

            # update parameters
            L = self.update(L, dAConv, dbConv, dAFC, dbFC)
            self.save()

            print('forward time %.3f, backward time %.3f, loss %.3f ' % \
                (middle - start, end - middle, L))

        time_end = time()
        print('training done, total time consumption %.3f' % (time_end - time_begin))

    def forward(self, N):
        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc
        B = self.B
        G = N // B # number of images in each batch
        self.G = G

        XB = self.XB

        # define forward funcion for spark map
        def forward_map(batch):
            start = batch * G
            end = start + G

            X = XB.value[start:end, :, :, :]
            R1 = conv.forward(X)
            X = None

            R2 = relu.forward(R1)
            # save R1
            save_matrix('R1_batch_' + str(batch), R1)
            R1 = None

            R3 = pool.forward(R2)
            # save R2
            save_matrix('R2_batch_' + str(batch), R2)
            R2 = None

            R4 = fc.forward(R3)
            # save R3
            save_matrix('R3_batch_' + str(batch), R3)
            R3 = None

            return [batch, R4]

        def forward_reduce(a, b):
            batches = np.append(a[0], b[0])
            R4 = np.append(a[1], b[1], 0)
            return [batches, R4]

        R = sc.parallelize(range(B), B).map(forward_map).reduce(forward_reduce)
        return R[0], R[1]

    def backward(self, batches, dS):
        # backward
        B = self.B
        G = self.G

        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc

        XB = self.XB

        def backward_map(pair):
            b = pair[0]
            dS = pair[1]

            # load R3
            R3 = load_matrix('R3_batch_' + str(b))
            dXFC, dAFC, dbFC = fc.backward(dS, R3)
            R3 = None

            # load R2
            R2 = load_matrix('R2_batch_' + str(b))
            dXPool = pool.backward(dXFC, R2)
            R2 = None

            # load R1
            R1 = load_matrix('R1_batch_' + str(b))
            dXReLU = relu.backward(dXPool, R1)
            R1 = None

            # load X
            start = b * G
            end = start + G
            X = XB.value[start:end, :, :, :]
            dXConv, dAConv, dbConv = conv.backward(dXReLU, X)
            X = None

            return [dAConv, dbConv, dAFC, dbFC]

        def backward_reduce(a, b):
            return np.sum([a, b], 0)

        # construct collection for map reduce
        pairs = []
        for i in range(0, len(batches)):
            b = batches[i]
            dS_b = dS[b * G:b * G + G, :]
            pairs.append([b, dS_b])

        R = sc.parallelize(pairs, len(pairs)).map(backward_map).reduce(backward_reduce)
        dAConv = R[0]
        dbConv = R[1]
        dAFC = R[2]
        dbFC = R[3]
        return dAConv, dbConv, dAFC, dbFC
