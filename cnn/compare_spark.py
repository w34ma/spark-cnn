import numpy as np
from cnn import CNN
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *
from time import time

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

RS = cnn.forward(X)
R1, R2, R3, R4 = RS
L, dS = softmax(R4, Y)

spark_start = time()

dXFC_2 = []
dAFC_2 = []
dbFC_2 = []
dXPool_2 = []
dXReLU_2 = []
dXConv_2 = []
dAConv_2 = []
dbConv_2 = []

step = 100

for i in range(0, N // step):
    dS_i = dS[step * i : step * i + step, :].reshape(step, C)
    # R3_i = R3[step * i : step * i + step, :, :, :].reshape(step, W // 2, N // 2, K)
    R3_i = R3[step * i : step * i + step, :, :, :]
    R2_i = R2[step * i : step * i + step, :, :, :]
    R1_i = R1[step * i : step * i + step, :, :, :]
    X_i = X[step * i : step * i + step, :, :, :]
    # fc
    dXFC_i, dAFC_i, dbFC_i = cnn.fc.backward(dS_i, R3_i)
    dXFC_2.append(dXFC_i)
    dAFC_2.append(dAFC_i)
    dbFC_2.append(dbFC_i)

    # pool
    dXPool_i = cnn.pool.backward(dXFC_i, R2_i)
    dXPool_2.append(dXPool_i)

    # relu
    dXReLU_i = cnn.relu.backward(dXPool_i, R1_i)
    dXReLU_2.append(dXReLU_i)

    # conv
    dXConv_i, dAConv_i, dbConv_i = cnn.conv.backward(dXReLU_i, X_i)
    dXConv_2.append(dXConv_i)
    dAConv_2.append(dAConv_i)
    dbConv_2.append(dbConv_i)


dXFC_2 = np.concatenate(dXFC_2)
dAFC_2 = np.sum(dAFC_2, 0)
dbFC_2 = np.sum(dbFC_2, 0)

dXPool_2 = np.concatenate(dXPool_2)

dXReLU_2 = np.concatenate(dXReLU_2)

dXConv_2 = np.concatenate(dXConv_2)
dAConv_2 = np.sum(dAConv_2, 0)
dbConv_2 = np.sum(dbConv_2, 0)

spark_end = time()
print('Spark: %.3f' % (spark_end - spark_start))

"""
assert np.allclose(dXFC_1, dXFC_2), 'dX failed'
assert np.allclose(dAFC_1, dAFC_2), 'dAFC failed'
assert np.allclose(dbFC_1, dbFC_2), 'dbFC failed'

assert np.allclose(dXPool_1, dXPool_2), 'dXPool failed'

assert np.allclose(dXReLU_1, dXReLU_2), 'dXReLU failed'

assert np.allclose(dXConv_1, dXConv_2), 'dXConv failed'
assert np.allclose(dAConv_1, dAConv_2), 'dAConv failed'
assert np.allclose(dbConv_1, dbConv_2), 'dbConv failed'
"""

print('done!')
