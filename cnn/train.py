import sys
from utils import *
from cnn import CNN

def run(size):
    # I: number of iterations
    # B: number of batches per iteration
    I = 10
    B = 10
    classifications, X_train, Y_train, X_test, Y_test = load_data()
    cnn = CNN(I, B)
    cnn.train(X_train, Y_train, classifications, size)

if __name__ == '__main__':
    size = 2000
    if len(sys.argv) > 1:
        size = int(sys.argv[1])
    run(size)
