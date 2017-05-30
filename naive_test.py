# test the performance of learnt CNN on its corresponding training data
import sys
import numpy as np
from spark.utils import *
from spark.cnn import CNN
from time import time

def test(size):
    print('Testing naive CNN for %d testing images' % (size))
    start = time()
    cnn = CNN(0)
    X, Y = load_testing_data(0, size)
    P = cnn.predict(X)
    P = np.argmax(P, 1)
    print('Predicted Classifications:')
    print(P)
    print('Correct Classifications:')
    print(Y)

    C = np.concatenate([P, Y]).reshape(2, -1).T
    C = [x for x in C if x[0] == x[1]]
    print('Correct:')
    print('%d/%d' % (len(C), size))
    end = time()
    print('Total time consumption: %.3f' % (end - start))

if __name__ == '__main__':
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    test(size)
