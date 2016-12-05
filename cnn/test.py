# test the performance of learnt CNN on test data
import sys
import numpy as np
from utils import *
from cnn import CNN
from time import time

def test(size):
    assert size <= 10000, 'we only have 10000 test data'
    start = time()
    X, Y = load_testing_data(0, size)
    cnn = CNN(0)
    P = cnn.predict(X)
    P = np.argmax(P, 1)
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
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    test(size)
