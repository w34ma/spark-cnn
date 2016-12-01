import sys
from utils import *
from spark_cnn import SparkCNN


def run(size, iteration):
    scnn = SparkCNN(iteration, 1)
    scnn.train(size)

if __name__ == '__main__':
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 10
    run(size, iteration)
