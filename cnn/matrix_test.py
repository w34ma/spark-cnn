# test whether cython implementation of im2col and col2im works properly

import numpy as np
from matrix import *

X = [
    1, 2, 1, 0, 0,
    2, 0, 0, 0, 2,
    1, 0, 2, 1, 0,
    0, 2, 0, 2, 1,
    2, 0, 0, 2, 0
]
X = np.asarray(X).reshape(1, 5, 5, 1)

print('im2col')
print('input')
print(X)
XC = im2col(X, 3, 2, 1)
print('output')
print(XC)

print('col2im')
N, W, H, D = X.shape
print('input')
print(XC)
X = col2im(XC, N, W, H, D, 3, 2, 1)
print('output')
print(X)
