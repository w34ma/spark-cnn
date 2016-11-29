import numpy as np
from matrix import *

X = np.arange(1 * 5 * 5 * 3).reshape(1, 5, 5, 3)
XC = im2col(X, 3, 2, 1)
print(XC)
