# test the performance of learnt CNN on its corresponding training data
import sys
import numpy as np
from utils import *
from cnn import CNN

def test(size):
    X, Y = load_training_data(0, size)
    print(X.shape)
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
    print(len(C))


if __name__ == '__main__':
    size = 2000
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    test(size)
