# spark enabled CNN
from pyspark.sql import SparkSession

import numpy as np
from time import time
from spark_conv import SparkConvolutionLayer
from relu import ReLULayer
from pool import PoolingLayer
from fc import FCLayer
from utils import *

from cnn import CNN

class SparkCNN(CNN):
    def __init__(self, I, B):
        CNN.__init__(self, I)
        self.B = B # number of batches
        # create spark context
        spark = SparkSession.builder.appName('cnn').getOrCreate()
        self.sc = spark.sparkContext

    """ override init_layers with spark """
    def init_layers(self, C):
        # initialize layers
        self.conv = SparkConvolutionLayer(32, 3, 1, 3, 1, self.sc, 4)
        self.relu = ReLULayer()
        self.pool = PoolingLayer(2, 2)
        self.fc = FCLayer(16, 16, 32, C)
