# spark enabled CNN with locality
from pyspark.sql import SparkSession
import os
import numpy as np
from time import time
from spark.conv import ConvolutionLayer
from spark.relu import ReLULayer
from spark.pool import PoolingLayer
from spark.fc import FCLayer
from spark.utils import *
from spark.spark_cnn import SparkCNN

class LocalityCNN(SparkCNN):
    def __init__(self, I, B):
        SparkCNN.__init__(self, I, B)
        self.name = 'locality_cnn'
        self.B = B # number of batches
        # create spark context
        spark = SparkSession.builder.appName('locality-cnn').getOrCreate()
        self.sc = spark.sparkContext

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

            # clear batch files
            clear_batches()

            # forward
            start = time()
            R4 = self.forward(size)
            middle = time()

            # calculate loss and gradients
            L, dS = softmax(R4, Y)

            dAConv, dbConv, dAFC, dbFC = self.backward(dS)
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
            # save X
            X = None

            R2 = relu.forward(R1)
            # save R1
            key_R1 = save_matrix_redis('R1_batch_' + str(batch), R1)
            R1 = None

            R3 = pool.forward(R2)
            # save R2
            key_R2 = save_matrix_redis('R2_batch_' + str(batch), R2)
            R2 = None

            R4 = fc.forward(R3)
            # save R3
            key_R3 = save_matrix_redis('R3_batch_' + str(batch), R3)
            R3 = None

            # save batch numbers to take advantage of locality
            secret = '{0}#{1}#{2}#{3}'.format(batch, key_R1, key_R2, key_R3)
            save_batch(secret)

            return R4

        def forward_reduce(a, b):
            return np.append(a, b, 0)

        R4 = sc.parallelize(range(B), B).map(forward_map).reduce(forward_reduce)
        return R4

    def backward(self, dS):
        # backward
        B = self.B
        G = self.G

        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc

        XB = self.XB
        # first broadcast dS to all nodes
        begin = time()
        dSB = sc.broadcast(dS)
        end = time()
        print('Backward broadcasting done time %.4f' % (end - begin))

        def backward_map(batch):
            secret = batch[1]
            b, key_R1, key_R2, key_R3 = secret.split('#')
            b = int(b)
            start = b * G
            end = start + G

            dS = dSB.value[start:end, :]

            # load R3
            R3 = load_matrix_redis(key_R3)
            dXFC, dAFC, dbFC = fc.backward(dS, R3)
            R3 = None

            # load R2
            R2 = load_matrix_redis(key_R2)
            dXPool = pool.backward(dXFC, R2)
            R2 = None

            # load R1
            R1 = load_matrix_redis(key_R1)
            dXReLU = relu.backward(dXPool, R1)
            R1 = None

            # load X

            X = XB.value[start:end, :, :, :]
            dXConv, dAConv, dbConv = conv.backward(dXReLU, X)
            X = None

            return [dAConv, dbConv, dAFC, dbFC]

        def backward_reduce(a, b):
            return np.sum([a, b], 0)

        # create RDD from hdfs directory
        R = sc.wholeTextFiles(get_hdfs_address_spark() + '/batches') \
            .map(backward_map).reduce(backward_reduce)

        dAConv = R[0]
        dbConv = R[1]
        dAFC = R[2]
        dbFC = R[3]
        end = time()
        return dAConv, dbConv, dAFC, dbFC
