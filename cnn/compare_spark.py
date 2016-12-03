from pyspark.sql import SparkSession

import numpy as np
from cnn import CNN
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *
from time import time

from queue import Queue
from threading import Thread

B = 100

N = 2000
W = 32
H = 32
D = 3
C = 10

K = 32

cnn = CNN(0)
cnn.conv = ConvolutionLayer(K, 3, 1, 3, 1)
cnn.relu = ReLULayer()
cnn.pool = PoolingLayer(2, 2)
cnn.fc = FCLayer(16, 16, K, C)

X = np.arange(N * W * H * D).reshape(N, W, H, D)
Y = np.ones(N * 1, np.int).reshape(N, 1)

print('before')
print(memory())
RS = cnn.forward(X)
R1, R2, R3, R4 = RS
L, dS = softmax(R4, Y)
print(memory())
print('after')

"""
# non spark
non_spark_start = time()
# fc
dXFC_1, dAFC_1, dbFC_1 = cnn.fc.backward(dS, R3)

# pool
dXPool_1 = cnn.pool.backward(dXFC_1, R2)

# ReLU
dXReLU_1 = cnn.relu.backward(dXPool_1, R1)

# conv
dXConv_1, dAConv_1, dbConv_1 = cnn.conv.backward(dXReLU_1, X)
non_spark_end = time()
print('Non Spark: %.3f' % (non_spark_end - non_spark_start))
"""

spark_start = time()

spark = SparkSession.builder.appName('cnn').getOrCreate()
sc = spark.sparkContext

def split(dS, R3, R2, R1, X):
    assert N % B == 0, 'invalid B'
    step = N // B

    batches = []
    for i in range(0, B):
        start = i * step
        end = start + step
        dS_i = dS[start:end, :]
        R3_i = R3[start:end, :, :, :]
        R2_i = R2[start:end, :, :, :]
        R1_i = R1[start:end, :, :, :]
        X_i = X[start:end, :, :, :]
        batches.append([dS_i, R3_i, R2_i, R1_i, X_i])

    return batches


def backward(batch):
    dS_b = batch[0]
    R3_b = batch[1]
    R2_b = batch[2]
    R1_b = batch[3]
    X_b = batch[4]

    dXFC_b, dAFC_b, dbFC_b = cnn.fc.backward(dS_b, R3_b)
    dXPool_b = cnn.pool.backward(dXFC_b, R2_b)
    dXReLU_b = cnn.relu.backward(dXPool_b, R1_b)
    dXConv_b, dAConv_b, dbConv_b = cnn.conv.backward(dXReLU_b, X_b)

    return [dAFC_b, dbFC_b, dAConv_b, dbConv_b]

split_start = time()
batches = sc.broadcast(split(dS, R3, R2, R1, X))
split_end = time()
print('spliting cost %.3f' % (split_end - split_start))

R = sc.parallelize(batches.value, 16).map(backward).reduce(collect)

def collect(a, b):
    return np.sum([a, b], 0)


"""

R = sc.parallelize(batches).map(backward).cache().reduce(collect)

dAFC = R[0]
dbFC = R[1]
dAConv = R[2]
dbConv = R[3]

spark_end = time()
print('Spark: %.3f' % (spark_end - spark_start))
"""


"""
assert np.allclose(dAFC_1, dAFC), 'dAFC failed'
assert np.allclose(dbFC_1, dbFC), 'dbFC failed'
assert np.allclose(dAConv_1, dAConv), 'dAConv failed'
assert np.allclose(dbConv_1, dbConv), 'dbConv failed'
"""

print('done!')
