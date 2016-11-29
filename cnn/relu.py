# Rectified Linear Unit Layer
import numpy as np

class ReLULayer():
    def forward(self, S):
        # input S: scores for C classifications for N images [N x K]
        # output: ReLUed scores [N x K]
        return np.maximum(0, S)
    def backward(self, df, S):
        # input: df are gradients from upstream [N x K]
        # input: S are input scores to forward function [N x K]
        # output: if input cells is less than 0 set the gradient to 0 [N x K]
        df[S <= 0] = 0
        return df
