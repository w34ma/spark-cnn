from matrix import *
from time import time

# pooling layer, using max pooling
class PoolingLayer():
    def __init__(self, F, S):
        self.F = F # size of filters
        self.S = S # stride of filters

    def forward(self, X):
        # input X activation matrix from conv layer [N x W x H x D]
        # where D is the depth, also the number of filters from conv layer
        # output max pooled activation matrix [N x W_ x H_ x D]

        F = self.F
        S = self.S

        N, W, H, D = X.shape
        assert (W - F) % S == 0 and (H - F) % S == 0, \
            'incompatible filter settings'

        W_ = (W - F) // S + 1
        H_ = (H - F) // S + 1

        # XC: (N * D * W_ * H_) * (F * F)
        # [(N x D) x W x H x 1] => [(N x D x W_ x H_) x (F x F)] 
        im2col_start = time()
        XC = im2col(X.transpose(0, 3, 1, 2).reshape(N * D, 1, W, H).transpose(0, 2, 3, 1), F, S, 0)
        im2col_end = time()
        print('--pool forward im2col done: time %.3f' % (im2col_end - im2col_start))

        argmax_start = time()
        XI = np.argmax(XC, axis = 1)
        argmax_end = time()
        print('--pool forward argmax done: time %.3f' % (argmax_end - argmax_start))

        transpose_start = time()
        R = XC[np.arange(XC.shape[0]), XI].reshape(N, D, W_, H_).transpose(0, 2, 3, 1)
        transpose_end = time()
        print('--pool forward matrix transformation done: time %.3f' % (transpose_end - transpose_start))

        print('pool forward at least memory used: %dMB' % ((XC.nbytes + XI.nbytes + R.nbytes) // 1024 // 1024))

        return R

    def backward(self, df, X):
        # input df are gradients from upstream [N x W_ x H_ x D]
        # input X is the activation matrix taken in at forward run [N x W x H x D]
        # output dX gradients on X [N x W x H x D]

        F = self.F
        S = self.S

        N, W_, H_, D = df.shape
        _, W, H, _ = X.shape

        
        # XC: (N * D * W_ * H_) * (F * F)
        im2col_start = time()
        XC = im2col(X.transpose(0, 3, 1, 2).reshape(N * D, 1, W, H).transpose(0, 2, 3, 1), F, S, 0)
        im2col_end = time()
        print('--pool backward im2col done: time %.3f' % (im2col_end - im2col_start))

        argmax_start = time()
        XI = np.argmax(XC, axis = 1)
        argmax_end = time()
        print('--pool backward argmax done: time %.3f' % (argmax_end - argmax_start))

        t1 = time()
        dX_col = np.zeros((N * D * W_ * H_, F * F))
        dX_col[np.arange(XC.shape[0]), XI] = df.transpose(0, 3, 1, 2).flatten()
        t2 = time()
        print('--pool backward transformation done: time %.3f' % (t2 - t1))

        col2im_start = time()
        dX = col2im(dX_col, N * D, W, H, 1, F, S, 0).reshape(N, D, W, H).transpose(0, 2, 3, 1)
        col2im_end = time()
        print('--pool backward col2im done: time %.3f' % (col2im_end - col2im_start))

        print('pool backward at least memory used: %dMB' % ((XC.nbytes + dX_col.nbytes + X.nbytes) // 1024 // 1024))
        # print('pool backward at least memory used: %dMB' % ((dX.nbytes + dX_col.nbytes) // 1024 // 1024))


        return dX
