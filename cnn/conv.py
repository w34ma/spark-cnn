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

        im2col_start = time()
        XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        im2col_end = time()
        X = None
        print('--conv forward im2col done: time %.3f' % (im2col_end - im2col_start))

        dot_start = time()
        R = np.dot(XC, A.reshape(K, F * F * D).T) + b.T
        dot_end = time()
        print('--conv forward dot done: time %.3f' % (dot_end - dot_start))

        print('conv forward at least memory used: %dMB' % ((XC.nbytes + R.nbytes) // 1024 // 1024))
        XC = None

        
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

        im2col_start = time()
        XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        im2col_end = time()
        X = None
        print('--conv backward im2col done: time %.3f' % (im2col_end - im2col_start))

        # stretch gradients to [(N x W_ x H_) x (F x F x D)]
        # dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
        # then get gradients on X
        t1 = time()
        dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
        t2 = time()
        # dX = col2im(np.dot(df.reshape(-1, K), A.reshape(K, -1)), N, W, H, D, F, S, P)
        dX = col2im(dXC, N, W, H, D, F, S, P)
        t3 = time()
        # XC = im2col(X, F, S, P)
        # print(X.shape)
        # print(XC.shape)
        t4 = time()
        # print('--convo backward im2col: %.3f' % (t4 - t3))
        
        dA = np.dot(df.reshape(-1, K).T, XC).reshape(K, F, F, D)
        t5 = time()

        # stretch original input to calculate gradients on filters
        # XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        # dA = np.dot(df.reshape(-1, K).T, im2col(X, F, S, P)).reshape(K, F, F, D)
        db = np.sum(df, axis=(0, 1, 2)).reshape(K, 1)
        t6 = time()

        
        print('--conv backward np.dot: %.3f' % (t2 - t1))
        print('--conv backward col2im: %.3f' % (t3 - t2))
        
        print('--conv backward np.dot: %.3f' % (t5 - t4))
        print('--conv backward np.sum: %.3f' % (t6 - t5))
        
        print('conv backward at least memory used: %dMB' % ((dX.nbytes + df.nbytes + XC.nbytes) // 1024 // 1024)) #XC > dA,dB


        return dX, dA, db
