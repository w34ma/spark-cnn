from matrix import *

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
        XC = im2col(X.transpose(0, 3, 1, 2).reshape(N * D, 1, W, H).transpose(0, 2, 3, 1), F, S, 0)
        XI = np.argmax(XC, axis = 1)
        return XC[np.arange(XC.shape[0]), XI].reshape(N, D, W_, H_).transpose(0, 2, 3, 1)
    
    def backward(self, df, X):
        # input df are gradients from upstream [N x W_ x H_ x D]
        # input X is the activation matrix taken in at forward run [N x W x H x D]
        # output dX gradients on X [N x W x H x D]

        F = self.F
        S = self.S

        N, W_, H_, D = df.shape
        _, W, H, _ = X.shape

        dX_col = np.zeros((N * D * W_ * H_, F * F))
        # XC: (N * D * W_ * H_) * (F * F)
        XC = im2col(X.transpose(0, 3, 1, 2).reshape(N * D, 1, W, H).transpose(0, 2, 3, 1), F, S, 0)
        XI = np.argmax(XC, axis = 1)

        dX_col[np.arange(XC.shape[0]), XI] = df.transpose(0, 3, 1, 2).flatten() 
        dX = col2im(dX_col, N * D, W, H, 1, F, S, 0).reshape(N, D, W, H).transpose(0, 2, 3, 1)

        return dX

