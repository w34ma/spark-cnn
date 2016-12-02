from pyspark.sql import SparkSession

import numpy as np
from cnn import CNN
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *
from time import time

N = 1000
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

RS = cnn.forward(X)
R1, R2, R3, R4 = RS
L, dS = softmax(R4, Y)

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


step = 100
batches = N // step

def calculation(b):
    start = b * step
    end = start + step
    dS_b = dS[start:end, :]
    R3_b = R3[start:end, :, :, :]
    R2_b = R2[start:end, :, :, :]
    R1_b = R1[start:end, :, :, :]
    X_b = X[start:end, :, :, :]

    dXFC_b, dAFC_b, dbFC_b = cnn.fc.backward(dS_b, R3_b)
    dXPool_b = cnn.pool.backward(dXFC_b, R2_b)
    dXReLU_b = cnn.relu.backward(dXPool_b, R1_b)
    dXConv_b, dAConv_b, dbConv_b = cnn.conv.backward(dXReLU_b, X_b)

    return [dAFC_b, dbFC_b, dAConv_b, dbConv_b]


R = sc.parallelize(range(0, batches)).map(calculation).collect()
R = np.sum(np.asarray(R), 0)
dAFC = R[0]
dbFC = R[1]
dAConv = R[2]
dbConv = R[3]
print(dAFC.shape)
print(dbFC.shape)
print(dAConv.shape)
print(dbConv.shape)

spark_end = time()
print('Spark: %.3f' % (spark_end - spark_start))


"""
assert np.allclose(dAFC_1, dAFC), 'dAFC failed'
assert np.allclose(dbFC_1, dbFC), 'dbFC failed'
assert np.allclose(dAConv_1, dAConv), 'dAConv failed'
assert np.allclose(dbConv_1, dbConv), 'dbConv failed'
"""

print('done!')
