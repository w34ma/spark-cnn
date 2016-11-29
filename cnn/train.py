import sys
from utils import *
from cnn import CNN

def run(size):
    # I: number of iterations
    # B: number of batches per iteration
    I = 10
    cnn = CNN(I)
    cnn.train(size)

if __name__ == '__main__':
    size = 1000
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    run(size)
