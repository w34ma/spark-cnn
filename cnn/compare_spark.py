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

dS_shape = dS.shape
R3_shape = R3.shape
R2_shape = R2.shape
R1_shape = R1.shape
X_shape = X.shape

dS_flat = dS.reshape(N, -1)
R3_flat = R3.reshape(N, -1)
R2_flat = R2.reshape(N, -1)
R1_flat = R1.reshape(N, -1)
X_flat = X.reshape(N, -1)

dS_c = dS_flat.shape[1]
R3_c = dS_c + R3_flat.shape[1]
R2_c = R3_c + R2_flat.shape[1]
R1_c = R2_c + R1_flat.shape[1]
X_c = R1_c + X_flat.shape[1]

M = np.concatenate([dS_flat, R3_flat, R2_flat, R1_flat, X_flat], 1)

dS_flat = None
R3_flat = None
R2_flat = None
R1_flat = None
X_flat = None


def calculate(B):
    [dS_b, R3_b, R2_b, R1_b, X_b] = np.split(B, [dS_c, R3_c, R2_c, R1_c])
    dS_b = dS_b.reshape(1, dS_shape[1])
    R3_b = R3_b.reshape(1, R3_shape[1], R3_shape[2], R3_shape[3])
    R2_b = R2_b.reshape(1, R2_shape[1], R2_shape[2], R2_shape[3])
    R1_b = R1_b.reshape(1, R1_shape[1], R1_shape[2], R1_shape[3])
    X_b = X_b.reshape(1, X_shape[1], X_shape[2], X_shape[3])

    dXFC_b, dAFC_b, dbFC_b = cnn.fc.backward(dS_b, R3_b)
    dXPool_b = cnn.pool.backward(dXFC_b, R2_b)
    dXReLU_b = cnn.relu.backward(dXPool_b, R1_b)
    dXConv_b, dAConv_b, dbConv_b = cnn.conv.backward(dXReLU_b, X_b)

    return [dAFC_b, dbFC_b, dAConv_b, dbConv_b]

R = sc.parallelize(M, 4).map(calculate).collect()
# R = R.collect()
# R = np.sum(np.asarray(R), 0)

"""
R = np.sum(np.asarray(R), 0)

dAFC = R[0]
dbFC = R[1]
dAConv = R[2]
dbConv = R[3]
"""
spark_end = time()
print('Spark: %.3f' % (spark_end - spark_start))



"""
assert np.allclose(dAFC_1, dAFC), 'dAFC failed'
assert np.allclose(dbFC_1, dbFC), 'dbFC failed'
assert np.allclose(dAConv_1, dAConv), 'dAConv failed'
assert np.allclose(dbConv_1, dbConv), 'dbConv failed'
"""
print('done!')
