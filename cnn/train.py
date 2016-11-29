import sys
from utils import *
from cnn import CNN

def run(size, iteration):
    # I: number of iterations
    # B: number of batches per iteration
    cnn = CNN(iteration)
    cnn.train(size)

if __name__ == '__main__':
    size = 1000
    iteration = 10
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    if len(sys.argv) > 2:
        iteration = int(sys.argv[2])

    run(size, iteration)
