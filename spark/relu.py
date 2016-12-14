# Rectified Linear Unit Layer
import numpy as np

class ReLULayer():
    def forward(self, R):
        # input R: result from conv layer [N x W x H x K]
        # output: processed R, every element >= 0
        return np.maximum(0, R)
    def backward(self, df, R):
        # input: df are gradients from upstream [N x W x H x K]
        # input: R are input in the forward run [N x W x H x K]
        # output: if input cells is less than 0 set the gradient to 0
        df[R <= 0] = 0
        return df
