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
