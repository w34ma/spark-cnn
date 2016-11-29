# test the performance of learnt CNN on its corresponding training data

import numpy as np
from utils import *
from cnn import CNN

def test(size):
    cnn = CNN()


if __name__ == '__main__':
    size = 2000
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    test(size)
