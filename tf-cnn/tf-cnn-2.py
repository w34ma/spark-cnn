import tensorflow as tf
import numpy as np
import os
import glob

CIFAR10_PATH = '../cifar10-bin'
TRAIN_EPOCH_NUM = 50000
TEST_EPOCH_NUM = 10000

def read_data(files):
    class cifar10(object):
        pass
    result = cifar10()

    result.label_size = 1
    result.image_height = 32
    result.image_width = 32
    result.image_depth = 3
    result.image_size = image_height * image_width * image_depth
    result.entry_size = label_byte + image_size
    
    reader = tf.FixedLengthRecordReader(record_bytes = result.entry_size)
    result.key, value = reader.read(files)
    entry = tf.decode_raw(value, tf.unit8)
    result.label = tf.cast(tf.slice(entry, [0], [result.label_size]), tf.int32)
    result.image = tf.reshape(tf.slice(entry, [result.label_size], [result.image_size]), [result.image_depth, result.image_height, result.image_width])

    # convert from [depth, height, width] to [height, width, depth]
    result.image = tf.transpose(result.image, [1, 2, 0])
    result.image = tf.cast(result.image, tf.float32)

    return result

def generate_batch(image, label, batch_size, min_queue_size):
    thread_num = 16
    image, label = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads = thread_num, capacity = min_queue_size + 3 * batch_size, min_after_dequeue = min_queue_size)
    return image, tf.reshape(label, batch_size)

def train_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "data_batch_*.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = 0.4 * TRAIN_EPOCH_NUM 
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

def test_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "test_batch.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = 0.4 * TEST_EPOCH_NUM
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev = 0.1)
    return tf.Variable(initial)

def bias_variable(shape):
    initial = tf.constant(0.1, shape = shape)
    return tf.Variable(inital)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

def max_pool(x):
    return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') 

def build_model(x):
    conv_W = weight_variable([5, 5, 3, 64])
    conv_b = bias_variable([64])
    conv_out = conv2d(x, conv_W) + conv_b
    relu_out = tf.nn.relu(conv_out)
    pool_out = max_pool(relu_out)
    fc_W = weight_variable([1024, 10])
    fc_b = bias_variable([10])
    fc_out = tf.matmul(pool_out, fc_W) + fc_b
    return fc_out

def cal_loss(image, label):

def train_model(loss):

def train():
    x = tf.placeholder(tf.float32, shape = shape) 
    y = tf.placeholder(tf.float32, shape = shape)
    cnn_y = build_mode(x)
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cnn_y, y))    
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    corr_pred = tf.equal(tf.argmax(cnn_y, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    sess = tf.InteractiveSession()
    sess.run(tf.initialize_all_variables())
    for i in range(ITERATION)):
        image, label = train_input(64)
        if i % 100 == 0:
            train_acc = acc.eval(feed_dict = {x: image, y: label, keep_prob: 1.0}
            print train_acc
        train_step.run(feed_dict = {x:image, y: label, keep_prob: 0.5})

if __name__ = '__main__':
    train()
