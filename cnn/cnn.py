import numpy as np
from time import time
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *

class CNN():
    def __init__(self, I, B):
        self.I = I # I: number of iterations
        self.B = B # B: number of batches for each iteration

        # hyper parameters settings
        self.rho = 1e-2 # learning rate
        self.mu = 0.9 # momentum
        self.lam = 0.1 # regularization strength

    def train(self, X, Y, classifications):
        N, W, H, D = X.shape
        C = len(classifications)
        self.N = N # N: number of images
        self.W = W # W: width of each image
        self.H = H # H: height of each image
        self.D = D # D: depth of each image
        self.C = C # C: number of classifications

        assert N % self.B == 0, 'batch number cannot divide total number of images'
        batch = N // self.B

        # initialize layers
        self.conv = ConvolutionLayer(5, 5, 1, 3, 2)
        self.relu = ReLULayer()
        self.pool = PoolingLayer(5, 2)
        self.fc = FCLayer(16, 16, 5, C)

        print('Start training CNN...')
        # preprocess make all points between [-0.5, 0.5]
        X = X[0:1000, :, :, :] / 255.0 - 0.5
        for i in range(0, self.I):
            print('iteration %d:' % i)
            # forward
            start = time()
            RS = self.forward(X)
            middle = time()

            # backward
            L = self.backward(X, Y, RS)
            end = time()
            print('forward time %.3f, backward time %.3f, loss %.3f ' % \
                (middle - start, end - middle, L))

            """
            results = []
            for j in range(0, self.B):
                samples_start = j * batch
                samples_end = samples_start + batch
                samples = X[samples_start:samples_end,:,:,:]
                labels = Y[samples_start:samples_end]
                print('iteration %d batch %d, range [%d, %d)' % \
                    (i + 1, j + 1, samples_start, samples_end))

                #forward
                start = time()
                RS = self.forward(samples) # get forward results
                middle = time()

                #backward
                loss = self.backward(samples, labels, RS)
                end = time()
                print('forward time %.3f, backward time %.3f ' % \
                    (middle - start, end - middle))
            """

    def forward(self, X):
        # X are the images [N x W x H x D]
        start = time()
        R1 = self.conv.forward(X) # result from first conv layer
        end = time()
        print('layer conv forward done: time %.3f' % (end - start))

        start = time()
        R2 = self.relu.forward(R1)
        end = time()
        print('layer relu forward done: time %.3f' % (end - start))

        start = time()
        R3 = self.pool.forward(R2)
        end = time()
        print('layer pool forward done: time %.3f' % (end - start))

        start = time()
        R4 = self.fc.forward(R3)
        end = time()
        print('layer fc forward done: time %.3f' % (end - start))

        return [R1, R2, R3, R4]

    def backward(self, X, Y, RS):
        # X are the images [N x W x H x D]
        # Y are the correct labels [N x 1]
        # RS are results from forward run
        R1, R2, R3, R4 = RS

        start = time()
        L, dS = softmax(R4, Y) # get loss and gradients with softmax function
        end = time()
        print('softmax loss calculation backward done: time %.3f' % (end - start))

        start = time()
        dX, dAFC, dbFC = self.fc.backward(dS, R3)
        end = time()
        print('layer fc backward done: time %.3f' % (end - start))

        start = time()
        dX = self.pool.backward(dX, R2)
        end = time()
        print('layer pool backward done: time %.3f' % (end - start))

        start = time()
        dX = self.relu.backward(dX, R1)
        end = time()
        print('layer relu backward done: time %.3f' % (end - start))

        start = time()
        dX, dAConv, dbConv = self.conv.backward(dX, X)
        end = time()
        print('layer conv backward done: time %.3f' % (end - start))

        # regularization
        L += 0.5 * self.lam * np.sum(self.conv.A * self.conv.A)
        L += 0.5 * self.lam * np.sum(self.fc.A * self.fc.A)
        dAFC += self.lam * self.fc.A
        dAConv += self.lam * self.conv.A

        # update neural network parameters
        # velocities
        self.conv.V = self.mu * self.conv.V - self.rho * dAConv
        self.fc.V = self.mu * self.fc.V - self.rho * dAFC
        # weights
        self.conv.A += self.conv.V
        self.fc.A += self.fc.V
        # biases
        self.conv.b += (0 - self.rho) * dbConv
        self.fc.b += (0 - self.rho) * dbFC

        return L
