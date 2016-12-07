# test the performance of learnt CNN on its corresponding training data
import sys
import numpy as np
from utils import *
from spark_cnn import SparkCNN
from time import time

def test(size, batches):
    start = time()
    cnn = SparkCNN(0, batches)
    P, Y = cnn.predict(size)
    P = np.argmax(P, 1)
    print('Batches: %d' % batches)
    print('Prediction:')
    print(P)
    print('Answer:')
    print(Y)

    C = np.concatenate([P, Y]).reshape(2, -1).T
    C = [x for x in C if x[0] == x[1]]
    print('Correct:')
    print('%d/%d' % (len(C), size))
    end = time()
    print('Total time consumption: %.3f' % (end - start))

if __name__ == '__main__':
    size = 2000
    batches = 4
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) > 2:
        batches = int(sys.argv[2])
    test(size, batches)
