import numpy as np
from time import time
from pyspark.sql import SparkSession

# images
N = 10
W = 32
H = 32
D = 3

# filter
K = 64
F = 3
S = 1
D = 3
P = 1

# after filter
W_ = (W - F + 2 * P) // S + 1
H_ = (H - F + 2 * P) // S + 1

# simulation of convolution layer forward matrix multiplication

random_start = time()
M1 = np.random.randn(N * W_ * H_, F * F * D)
M2 = np.random.randn(F * F * D, K)
random_end = time()
print('Creating matrices cost: %.4f' % (random_end - random_start))


# with numpy matrix multiplication
print('Numpy starts')
np_start = time()
R = np.dot(M1, M2)
np_end = time()
print('Numpy ends')
print('Numpy time cost: %.4f' % (np_end - np_start))

rowM1 = len(M1)
colM1 = len(M1[0])
rowM2 = len(M2)
colM2 = len(M2[0])


"""
# navie matrix multiplication
print('Navie starts')
navie_start = time()

rowM1 = len(M1)
colM1 = len(M1[0])
rowM2 = len(M2)
colM2 = len(M2[0])

newM = [[0 for row in range(colM2)] for col in range(rowM1)]

for i in range(rowM1):
    for j in range(colM2):
        for k in range(colM1):
            newM[i][j] += M1[i][k] * M2[k][j]

navie_end = time()
print('Naive ends')
print('Naive time cost: %.4f' % (navie_end - navie_start))
"""
# spark matrix multiplication - inner and outter product
print('Spark starts')
spark_start = time()

spark = SparkSession.builder.appName('cnn').getOrCreate()
sc = spark.sparkContext

a = []
M1T = M1.T
for i in range(colM1):
    t = [M1T[i], M2[i]]
    a.append(t)


def multiplication(t):
    column = t[0].reshape(-1, 1)
    row = t[1].reshape(1, -1)
    result = np.dot(column, row)
    return result

def summation(a, b):
    return np.sum([a, b], (0));

result_spark = sc.parallelize(a).map(multiplication).reduce(summation)
spark_end = time()
print('Spark ends')
print('Spark time cost: %.4f' % (spark_end - spark_start))

assert np.allclose(result_spark, R), 'failed'

