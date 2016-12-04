# spark enabled convolution layer
import numpy as np
from matrix import *
from conv import ConvolutionLayer

class SparkConvolutionLayer(ConvolutionLayer):
    def __init__(self, K, F, S, D, P, sc, partitions):
        ConvolutionLayer.__init__(self, K, F, S, D, P)
        self.sc = sc # getting spark context
        self.partitions = partitions
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
        df_to_split = df.reshape(-1, K)
        A_to_split = A.reshape(K, -1)
        dX = []

        def f(k):
            df_k = df_to_split[:, k].reshape(-1, 1)
            A_k = A_to_split[k, :].reshape(1, -1)
            return col2im(np.dot(df_k, A_k), N, W, H, D, F, S, P)

        dX = self.sc.parallelize(range(0, K), self.partitions).map(f).collect()
        dX = np.sum(np.asarray(dX), (0))

        # dX = col2im(np.dot(df.reshape(-1, K), A.reshape(K, -1)), N, W, H, D, F, S, P)
        # stretch original input to calculate gradients on filters
        # XC = im2col(X, F, S, P) # [(N x W_ x H_) x (F x F x D)]
        dA = np.dot(df.reshape(-1, K).T, im2col(X, F, S, P)).reshape(K, F, F, D)
        db = np.sum(df, axis=(0, 1, 2)).reshape(K, 1)

        return dX, dA, db
