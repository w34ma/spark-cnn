# convolution layer
import numpy as np
import math
from matrix import *

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
        self.b = np.zeros((K, 1))

    def forward(self, X):
        K = self.K
        F = self.F
        S = self.S
        P = self.P
        A = self.A
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

        XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        R = np.dot(XC, A.reshape(K, F * F * D).T) + b
        return R.reshape(N, W_, H_, K)

    def backward(self, df, X):
        # input: df are gradients from upstream [N x W_ x H_ x K]
        # input: X is a [N x W x H x D] matrix of images
        # output: dX gradient on X [N x W x H x D]
        #         dA gradient on A [K x F x F x D]
        #         db gradient on b [K]
        K = self.K
        F = self.F
        S = self.S
        P = self.P
        A = self.A # [K x F x F x D]

        N, K, W_, H_ = df.shape
        _, F, _, D = A.shape

        # stretch gradients to [(N x W_ x H_) x (F x F x D)]
        dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
        # get gradients on X
        dX = col2im(dXC, X.shape, F, S, P)

        # stretch original input to calculate gradients on filters
        XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        dA = np.dot(df.reshape(-1, K).T, XC).reshape(K, F, F, D)
        db = np.sum(df, axis=(0, 1, 2)).reshape(K, 1)

        return dX, dA, db
