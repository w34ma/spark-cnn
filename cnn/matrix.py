import numpy as np

# get width, height, depth combination points
# looping all combinations of W_ x H_ x F x F x D
# at every filter matrix
def computePoints(W_, H_, F, D):
    ws = []
    hs = []
    ds = []
    for w in range(0, W_):
        for h in range(0, H_):
            for fw in range(0, F):
                for fh in range(0, F):
                    for d in range(0, D):
                        ws.append(w * S + fw)
                        hs.append(h * S + fh)
                        ds.append(d)
    return ws, hs, ds

# matrix calculation helpers
def im2col(X, F, S, P):
    # convert matrix to stretched columns
    # output [(N x W_ x H_) x (F x F x D)]
    # X: input matrix, of size [N x W x H x D]
    # F: filter size (assume same width and height)
    # S: filter stride
    # P: zero paddings
    N, W, H, D = X.shape
    # compute output matrix dimension
    W_ = (W - F + 2 * P) // S + 1
    H_ = (H - F + 2 * P) // S + 1

    # pad X with zeros and move N to last column
    X0 = np.pad(X, ((0, 0), (P, P), (P, P), (0, 0)), 'constant')

    ws, hs, ds = computePoints(W_, H_, F, D)
    return X0[:, ws, hs, ds].reshape(N * W_ * H_, F * F * D)

def col2im(dXC, shape, F, S, P):
    # convert gradients on stretched column matrix back to original matrix
    # output [N x W x H x D]
    # dXC: gradients [(F x F x D) x (W_ x H_ x N)]
    # F: filter size (assume same width and height)
    # S: filter stride
    # P: zero paddings
    N, W, H, D = shape
    W_ = (W - F + 2 * P) // S + 1
    H_ = (H - F + 2 * P) // S + 1

    ws, hs, ds = computePoints(W_, H_, F, D)
    # initialize dX with zero paddings to all zeros
    dX0 = np.zeros((N, W + 2 * P, H + 2 * P, D))

    # add gradients from all filter matrix in stretched matrix
    # back to original matrix (with zero paddings)
    np.add.at(dX0, (slice(None), ws, hs, ds)), dXC.reshape(N, -1))

    if P > 0:
        dX = dX0[:, P:-P, P:-P, :] # remove zero paddings
    else:
        dX = dX0

    return dX
