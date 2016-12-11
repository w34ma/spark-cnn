import sys
from spark.utils import *
from spark.locality_cnn import LocalityCNN

def run(size, iteration, batch):
    scnn = SparkCNN(iteration, batch)
    print('Training Locality CNN with %d iterations (%d batches) for %d images' %
          (iteration, batch, size))
    scnn.train(size)

if __name__ == '__main__':
    size = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    iteration = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    batch = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    run(size, iteration, batch)
