import sys
from spark.utils import *
from spark.cnn import CNN

def run(size, iteration):
    # I: number of iterations
    # B: number of batches per iteration
    cnn = CNN(iteration)
    print('Training naive CNN with %d iterations for %d images' % (iteration, size))
    cnn.train(size)

if __name__ == '__main__':
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    run(size, iteration)
