import numpy as np
import sys
sys.path.append('..')
sys.path.append('../cnn')

from cnn.matrix import *
from cnn.utils import *

if __name__ == '__main__':

    N = 1
    W_ = 3
    H_ = 3
    K = 2
    W = 5
    H = 5
    D = 3
    F = 3
    P = 1
    S = 2

    X = np.arange(0, N * W * H * D).reshape(N, W, H, D)
    df = np.arange(0, N * W_ * H_ * K).reshape(N, W_, H_, K)
    A = np.arange(0, K * F * F * D).reshape(K, F, F, D)
    dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
    dX = col2im(dXC, N, W, H, D, F, S, P)

    XC = im2col(X, F, S, P)
    dA = np.dot(df.reshape(-1, K).T, XC).reshape(K, F, F, D)
    db = np.sum(df, axis=(0, 1, 2)).reshape(K, 1)

    print('approach 1')
    # print(dX)
    print(dA)
    # print(db)

    dfR = df.reshape(-1, K) # [9 x 2]
    AR = A.reshape(K, -1) # [2 x 27]
    dX = []
    for k in range(0, K):
        dfK = dfR[:, k].reshape(-1, 1)
        AK = AR[k, :].reshape(1, -1)
        dXCK = np.dot(dfK, AK)
        dXK = col2im(dXCK, N, W, H, D, F, S, P)
        dX.append(dXK)

    dX = np.sum(dX, (0))
    print('approach 2')
    #print(dX)
