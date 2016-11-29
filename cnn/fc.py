# fully connected layer
import numpy as np
import math

class FCLayer():
    def __init__(self, W, H, D, C):
        self.W = W # input width
        self.H = H # input height
        self.D = D # input depth
        self.C = C # output number of possible classifications
        # initialize velocities
        self.V = np.zeros([W * H * D, C])
        # initialize weights and biases
        # [(W x H x D) x C]
        self.A = math.sqrt(2.0 / (W * H * D * C)) * np.random.randn(W * H * D, C)
        self.b = np.zeros([1, C])

    def forward(self, X):
        # input X activation matrix [N x W x H x D]
        # output classifications [N x C]
        N, W, H, D = X.shape
        return np.dot(X.reshape(N, -1), self.A) + self.b

    def backward(self, df, X):
        # input df are gradients from upstream [N x C]
        # input X are input activation matrix from forward run [N x W x H x D]
        # output dX gradients on X [N x W x H x D]
        # output dA gradients on A [(W x H x D) x C]
        # output db gradients on b [K]
        N, C = df.shape
        dX = np.dot(df, self.A.T).reshape(X.shape)
        dA = np.dot(X.reshape(N, -1).T, df)
        db = np.sum(df, 0)
        return dX, dA, db
