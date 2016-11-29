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
    
    N, C = P.shape

    # calculate probabilities [N x C]
    print(P)
    ps = np.exp(P - np.max(P, 1, None, True))
    # ps = ps / np.sum(ps, 1, None, None, True)
    
    print(ps)

    P = np.argmax(P, 1)
    print('Prediction:')
    print(P)
    print('Answer:')
    print(Y)


if __name__ == '__main__':
    size = 2000
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    test(size)
