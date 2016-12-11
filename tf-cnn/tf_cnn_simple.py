import tensorflow as tf
import numpy as np
import os
import glob
import random
import cPickle
import time
import psutil

CIFAR10_PATH = '../cifar10-bin'
TRAIN_EPOCH_NUM = 50000
TEST_EPOCH_NUM = 10000
ITERATION_NUM = 10000
LABEL_NUM = 10

FRAC_MIN_QUEUE_SIZE = 0.4
BATCH_SIZE = 1000
LABEL_SIZE = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(initial)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') 

def build_model(x):
    conv_W = weight_variable([5, 5, 3, 32])
    conv_b = bias_variable([32])
    conv_out = conv2d(x, conv_W) + conv_b
    relu_out = tf.nn.relu(conv_out)
    pool_out = max_pool(relu_out)
    pool_out = tf.reshape(pool_out, [-1, 16 * 16 * 32])
    fc_W = weight_variable([16 * 16 * 32, LABEL_NUM])
    fc_b = bias_variable([LABEL_NUM])
    fc_out = tf.matmul(pool_out, fc_W) + fc_b
    return fc_out

def read_data(type):

    class reader(object):
        
        def __init__(self, type):
            self.data = np.ndarray(shape = [0, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH])
            self.label = []
            self.size = 0
            if type == 'train':
                self.read_all_train()
            else:
                self.read_one_test()

        def get(self, size):
            index = random.sample(xrange(self.size), size)
            image = self.data[index, :]
            image = np.reshape(image, [size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH])
            label = np.zeros([size, LABEL_NUM])
            for i in xrange(size):
                label[i, self.label[index[i]]] = 1 
            return image, label

        def read_one_test(self):
            fo = open('../cifar10/test_batch', 'rb')
            dict = cPickle.load(fo)
            fo.close()
            self.data = np.float32(dict['data'])
            self.label = dict['labels']
            self.size = len(dict['labels'])
            print self.data.shape
            print self.size

        def read_all_train(self):
            files = glob.glob('../cifar10/data_batch_*')
            for file in files:
                fo = open(file, 'rb')
                dict = cPickle.load(fo)
                fo.close()
                self.data = np.concatenate((self.data, np.float32(dict['data'])), axis = 0) # 10000 * 3072 ndarray
                self.label += dict['labels'] # 10000 list
                self.size += len(dict['labels'])
            print self.data.shape
            print self.size

    result = reader(type)
    return result

def train(process):
    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]) 
    y = tf.placeholder(tf.float32, shape = [BATCH_SIZE, LABEL_NUM])
    cnn_y = build_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cnn_y, y))    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    corr_pred = tf.equal(tf.argmax(cnn_y, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    train_data = read_data('train')
    test_data = read_data('test')
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        print "start training"
        for i in range(ITERATION_NUM):
            start = time.time()
            image, label = train_data.get(BATCH_SIZE)
            train_step.run(feed_dict = {x: image, y: label})
            dt = time.time() - start
            print("time for iteration %d is %f" % (i, dt))
            mem = (process.get_memory_info()[0] / float(2 ** 20))
            print("memory usage %f MB: " % mem)
            if i % 10 == 0:
                print("Now it's the %d round " % i)
                im, lb = test_data.get(BATCH_SIZE)
                train_acc = acc.eval(feed_dict = {x: im, y: lb})
                print("accuracy now is : %f " % train_acc)

if __name__ == '__main__':
    process = psutil.Process(os.getpid())
    train(process)
