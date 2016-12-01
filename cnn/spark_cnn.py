# spark enabled CNN

import numpy as np
from time import time
from conv import ConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *

from cnn import CNN

class SparkCNN(CNN):
    def __init__(self, I, B):
        CNN.__init__(self, I)
        self.B = B # number of batches

    """ override train with spark """
    def train(self, size = 1000):
        CNN.train(self, size)
