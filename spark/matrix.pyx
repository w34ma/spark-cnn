# matrix manipulation helpers im2col and col2im
import numpy as np
cimport numpy as np

# declare external C++ functions
cdef extern from "matrix_c.h":
    void im2col_c(double* X0, double* XC,
        int N, int W, int H, int D, int F, int S, int P)
    void col2im_c(double* dXC, double* dX0,
        int N, int W, int H, int D, int F, int S, int P)

def im2col(X, F, S, P):
    # convert matrix to stretched columns
    # X: input matrix, of size [N x W x H x D]
    # F: filter size (assume same width and height)
    # S: filter stride
    # P: zero paddings
    # output XC: [(N x W_ x H_) x (F x F x D)]

    N, W, H, D = X.shape
    # compute output matrix dimension
    W_ = (W - F + 2 * P) // S + 1
    H_ = (H - F + 2 * P) // S + 1

    # prepare input for C
    # zero padded X
    # conv zero padding
    X0 = np.pad(X, ((0, 0), (P, P), (P, P), (0, 0)), 'constant')
    X0 = np.require(X0, np.float64, ['C', 'A'])
    # prepare output for C
    # column stretched matrix
    XC = np.zeros([N * W_ * H_, F * F * D])
    XC = np.require(XC, np.float64, ['C', 'A'])
    # hand over calculation to C
    im2col_c(
        <double*> np.PyArray_DATA(X0),
        <double*> np.PyArray_DATA(XC),
        N, W, H, D, F, S, P
    )

    # print('most memory used in im2col: ' + memory())
    return XC

def col2im(dXC, N, W, H, D, F, S, P):
    # convert stretched columns back to matrix
    # dXC: gradients on stretched columns [(N x W_ x H_) x (F x F x D)]
    # N: number of images
    # W: width of output matrix
    # H: height of output matrix
    # D: depth of output matrix
    # F: filter size (assume same width and height)
    # S: filter stride
    # P: zero paddings
    # output dX: [N x W x H x D]

    # compute dimension of input gradient matrix without stretching
    W_ = (W - F + 2 * P) // S + 1
    H_ = (H - F + 2 * P) // S + 1

    # prepare input for C
    dXC = np.require(dXC, np.float64, ['C', 'A'])

    # prepare output for C (zero padded)
    dX0 = np.require(np.zeros([N, W + 2 * P, H + 2 * P, D]), np.float64, ['C', 'A'])

    # hand over calculation to C
    col2im_c(
        <double*> np.PyArray_DATA(dXC),
        <double*> np.PyArray_DATA(dX0),
        N, W, H, D, F, S, P
    )

    # remove zero paddings
    if P > 0:
        dX = dX0[:, P:-P, P:-P, :]
    else:
        dX = dX0

    # print('most memory used in col2im: ' + memory())
    return dX
