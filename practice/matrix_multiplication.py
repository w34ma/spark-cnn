import numpy as np
from matrix import *

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

df = np.arange(0, N * W_ * H_ * K).reshape(N, W_, H_, K)
A = np.arange(0, K * F * F * D).reshape(K, F, F, D)
dXC = np.dot(df.reshape(-1, K), A.reshape(K, -1))
dX = col2im(dXC, N, W, H, D, F, S, P)
