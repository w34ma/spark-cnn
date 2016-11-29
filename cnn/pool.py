# pooling layer, using max pooling
class PoolingLayer():
    def __init__(self, F, S):
        self.F = F # size of filters
        self.S = S # stride of filters
    def forward(self, X):
        # input X activation matrix from conv layer [N x W x H x D]
        # where D is the depth, also the number of filters from conv layer
        # output max pooled activation matrix [N x W_ x H_ x D]
    def backward(self, df, X):
        # input df are gradients from upstream [N x W_ x H_ x D]
        # input X is the activation matrix taken in at forward run [N x W x H x D]
        # output dX gradients on X [N x W x H x D]
