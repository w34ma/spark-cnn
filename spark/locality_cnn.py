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
        for i in range(0, self.I):
            print('iteration %d' % i)

            # clear batch files
            clear_batches()

            # forward
            start = time()
            R4, Y = self.forward(size)
            middle = time()

            # calculate loss and gradients
            L, dS = softmax(R4, Y)

            dAConv, dbConv, dAFC, dbFC = self.backward(dS)
            end = time()

            # update parameters
            L = self.update(L, dAConv, dbConv, dAFC, dbFC)

            print('forward time %.3f, backward time %.3f, loss %.3f ' % \
                (middle - start, end - middle, L))

        self.save()
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


        # define forward funcion for spark map
        def forward_map(batch):
            start = batch * G
            end = start + G

            X, Y = load_training_data(start, end)
            R1 = conv.forward(X)
            # save X
            save_matrix('X_batch_' + str(batch), X)
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

            # save batch numbers to take advantage of locality
            save_batch(batch)

            return [R4, Y]

        def forward_reduce(a, b):
            R4 = np.append(a[0], b[0], 0)
            Y = np.append(a[1], b[1], 0)
            return [R4, Y]

        R = sc.parallelize(range(B), B).map(forward_map).reduce(forward_reduce)
        return R[0], R[1]

    def backward(self, dS):
        # backward
        B = self.B
        G = self.G

        conv = self.conv
        relu = self.relu
        pool = self.pool
        fc = self.fc
        sc = self.sc

        # first broadcast dS to all nodes
        dS_broadcasted = sc.broadcast(dS)

        def backward_map(batch):
            b = int(batch[1])
            dS = dS_broadcasted.value[b * G:b * G + G, :]

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
            X = load_matrix('X_batch_' + str(b))
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
