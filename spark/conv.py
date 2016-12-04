# convolution layer
import numpy as np
import math
from time import time
from matrix import *
from utils import *

class ConvolutionLayer():
    def __init__(self, K, F, S, D, P):
        self.K = K # number of filters
        self.F = F # size of filters
        self.S = S # stride of filters
        self.D = D # depth of filters
        self.P = P # amount of zero paddings
        # initalize velocity
        self.V = np.zeros([K, F, F, D])
        # initialize weights and bias
        self.A = math.sqrt(2.0 / (K * F * F * D)) * np.random.randn(K, F, F, D)
        # self.A = math.sqrt(2.0 / (K * F * F * D)) * np.zeros((K, F, F, D))
        # self.A = 0.01 * np.random.randn(K, F, F, D)
        self.b = np.zeros((K, 1))

    def forward(self, X):
        K = self.K
        F = self.F
        S = self.S
        P = self.P
        A = self.A
        b = self.b
        # input: X is a [N x W x H x D] matrix of images
        # output: R: An activation matrix of size [N x W_ x H_ x K]
        # W_ = (W - F + 2P) / S + 1
        # H_ = (H - F + 2P) / S + 1
        N, W, H, D = X.shape
        assert D == self.D, "incompatible filter depth"
        assert (W - F + 2 * P) % S == 0 and (H - F + 2 * P) % S == 0, \
            'incompatible filter settings'

        # compute output matrix dimension
        W_ = (W - F + 2 * P) // S + 1
        H_ = (H - F + 2 * P) // S + 1

        # XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        R = np.dot(im2col(X, F, S, P), A.reshape(K, F * F * D).T) + b.T
        return R.reshape(N, W_, H_, K)

    def backward(self, df, X):
        # input: df are gradients from upstream [N x W_ x H_ x K]
        # input: X is a [N x W x H x D] matrix of images
        # output: dX gradient on X [N x W x H x D]
        #         dA gradient on A [K x F x F x D]
        #         db gradient on b [K]
        F = self.F
        S = self.S
        P = self.P
        A = self.A # [K x F x F x D]

        N, W_, H_, K = df.shape
        _, W, H, D = X.shape

        # stretch gradients to [(N x W_ x H_) x (F x F x D)]
        # dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
        # then get gradients on X
        # dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
        dX = col2im(np.dot(df.reshape(-1, K), A.reshape(K, -1)), N, W, H, D, F, S, P)
        # dX = col2im(dXC, N, W, H, D, F, S, P)
        XC = im2col(X, F, S, P)
        dA = np.dot(df.reshape(-1, K).T, im2col(X, F, S, P)).reshape(K, F, F, D)
        # dA = np.dot(df.reshape(-1, K).T, XC).reshape(K, F, F, D)

        # stretch original input to calculate gradients on filters
        # XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        # dA = np.dot(df.reshape(-1, K).T, im2col(X, F, S, P)).reshape(K, F, F, D)
        db = np.sum(df, axis=(0, 1, 2)).reshape(K, 1)
        t6 = time()

        """
        print('step 1: %.3f' % (t2 - t1))
        print('step 2: %.3f' % (t3 - t2))
        print('step 3: %.3f' % (t4 - t3))
        print('step 4: %.3f' % (t5 - t4))
        print('step 5: %.3f' % (t6 - t5))
        """

        return dX, dA, db
