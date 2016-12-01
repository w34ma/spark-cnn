import tensorflow as tf
import numpy as np
import os
import glob
import random
import cPickle
import time

CIFAR10_PATH = '../cifar10-bin'
TRAIN_EPOCH_NUM = 50000
TEST_EPOCH_NUM = 10000
ITERATION_NUM = 10000
LABEL_NUM = 10

FRAC_MIN_QUEUE_SIZE = 0.4
BATCH_SIZE = 64
LABEL_SIZE = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

def read_data(files):
    class cifar10(object):
        pass

    result = cifar10()
    result.label_size = LABEL_SIZE
    result.image_height = IMAGE_SIZE
    result.image_width = IMAGE_WIDTH
    result.image_depth = IMAGE_HEIGHT
    result.image_size = result.image_height * result.image_width * result.image_depth
    result.entry_size = result.label_size + result.image_size
    
    reader = tf.FixedLengthRecordReader(record_bytes = result.entry_size)
    result.key, value = reader.read(files)
    entry = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.slice(entry, [0], [result.label_size]), tf.int32)
    result.image = tf.reshape(tf.slice(entry, [result.label_size], [result.image_size]), [result.image_depth, result.image_height, result.image_width])

    # convert from [depth, height, width] to [height, width, depth]
    result.image = tf.transpose(result.image, [1, 2, 0])
    result.image = tf.cast(result.image, tf.float32)

    return result

def generate_batch(image, label, batch_size, min_queue_size):
    thread_num = 4
    image, label = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads = thread_num, capacity = min_queue_size + 3 * batch_size, min_after_dequeue = min_queue_size)
    label = tf.one_hot(tf.reshape(label, [batch_size]), LABEL_NUM)
    return image, label

def train_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "data_batch_*.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = 2000 
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

def test_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "test_batch.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = 2000
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

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

def read_train():

    class reader(object):
        
        def __init__(self):
            self.data = np.ndarray(shape = [0, IMAGE_WIDTH * IMAGE_HEIGHT * IMAGE_DEPTH])
            self.label = []
            self.size = 0

        def get(self, size):
            index = random.sample(xrange(self.size), size)
            image = self.data[index, :]
            image = np.reshape(image, [size, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH])
            label = np.zeros([size, LABEL_NUM])
            for i in xrange(size):
                label[i, self.label[index[i]]] = 1 
            return image, label

        def read_all(self):
            files = glob.glob('../cifar10/data_batch_*')
            for file in files:
                fo = open(file, 'rb')
                dict = cPickle.load(fo)
                fo.close()
                self.data = np.concatenate((self.data, dict['data']), axis = 0) # 10000 * 3072 ndarray
                self.label += dict['labels'] # 10000 list
                self.size += len(dict['labels'])

    result = reader()
    result.read_all()
    return result

def train():
    x = tf.placeholder(tf.float32, shape = [BATCH_SIZE, IMAGE_WIDTH, IMAGE_HEIGHT, IMAGE_DEPTH]) 
    y = tf.placeholder(tf.float32, shape = [BATCH_SIZE, LABEL_NUM])
    cnn_y = build_model(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cnn_y, y))    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    corr_pred = tf.equal(tf.argmax(cnn_y, 0), tf.argmax(y, 0))
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    train_data = read_train()
    sess = tf.Session()
    with sess.as_default():
        sess.run(tf.initialize_all_variables())
        print "start training"
        for i in range(ITERATION_NUM):
            image, label = train_data.get(BATCH_SIZE)
            if i % 10 == 0:
                print("Now it's the %d round \n" % i)
                train_step.run(feed_dict = {x: image, y: label})
                train_acc = acc.eval(feed_dict = {x: image, y: label})
                print(train_acc)

if __name__ == '__main__':
    train()
