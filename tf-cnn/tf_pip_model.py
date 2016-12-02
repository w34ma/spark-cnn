import tensorflow as tf
import numpy as np
import os
import glob
import time

CIFAR10_PATH = '../cifar10-bin'
CHECKPOINT_DIR = 'tf-backup'
TRAIN_EPOCH_NUM = 50000
TEST_EPOCH_NUM = 10000
ITERATION_NUM = 10000
LABEL_NUM = 10

FRAC_MIN_QUEUE_SIZE = 0.4
BATCH_SIZE = 32
LABEL_SIZE = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3


class cnn_model(object):

    def weight_variable(self, shape):
        initial = tf.truncated_normal(shape, stddev = 0.1)
        return tf.Variable(initial)

    def bias_variable(self, shape):
        initial = tf.constant(0.1, shape = shape)
        return tf.Variable(initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') 

    def init(self, x):
        conv_W = self.weight_variable([5, 5, 3, 32])
        conv_b = self.bias_variable([32])
        conv_out = self.conv2d(x, conv_W) + conv_b
        relu_out = tf.nn.relu(conv_out)
        pool_out = self.max_pool(relu_out)
        pool_out = tf.reshape(pool_out, [-1, 16 * 16 * 32])
        fc_W = self.weight_variable([16 * 16 * 32, LABEL_NUM])
        fc_b = self.bias_variable([LABEL_NUM])
        fc_out = tf.matmul(pool_out, fc_W) + fc_b
        return fc_out

def cal_loss(cnn_y, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cnn_y, y))    
    return loss

def train_model(loss):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    return train_step

def cal_acc(cnn_y, y):
    corr_pred = tf.equal(tf.argmax(cnn_y, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    return acc


