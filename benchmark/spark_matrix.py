# empirical study of matrix multiplication
from pyspark.sql import SparkSession
import numpy as np
from time import time

# initialize spark
spark = SparkSession.builder.appName('matrix-multiplication').getOrCreate()
sc = spark.sparkContext

N = 1000
W = 32
H = 32
K = 32
F = 5
D = 3

A = np.random.randn(N * W * H, K)
B = np.random.randn(K, F * F * D)


def numpy_matrix(A, B):
    print('numpy.dot starts')
    start = time()
    C = np.dot(A, B)
    end = time()
    print('numpy.dot ends, time %.4f' % (end - start))
    return C

def split_matrix(A, B, S):
    print('split_matrix starts')
    start = time()
    N = A.shape[0]
    step = N // S

    # send B to all nodes
    BB = sc.broadcast(B)
    AA = sc.broadcast(A)
    middle = time()
    print('broadcasting done, time %.4f' % (middle - start))

    def multiply(a):
        pos_start = a * step
        pos_end = pos_start + step
        return np.dot(AA.value[pos_start:pos_end, :], BB.value)

    def concat(a, b):
        return np.concatenate([a, b])

    C = sc.parallelize(range(S), S).map(multiply).reduce(concat)
    end = time()
    print('split_matrix ends, time %.4f' % (end - start))
    return C

def inout_matrix(A, B):
    print('inout_matrix starts')
    start = time()
    AT = A.T
    ts = []
    for i in range(len(AT)):
        ts.append([AT[i], B[i]])

    def multiply(t):
        col = t[0].reshape(-1,1)
        row = t[1].reshape(1,-1)
        return np.dot(col, row)

    def summation(a, b):
        return np.sum([a, b], (0))

    C = sc.parallelize(ts).map(multiply).reduce(summation)
    end = time()
    print('inout_matrix ends, time %.4f' % (end - start))
    return C

C1 = numpy_matrix(A, B)
C2 = split_matrix(A, B, 2)
C3 = inout_matrix(A, B)

assert np.allclose(C1, C2), 'failed C1 != C2'
assert np.allclose(C1, C3), 'failed C1 != C3'

print('done')
