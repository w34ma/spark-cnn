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

def build_mode(image):
    with tf.variable_scope('conv') as scope:
        kernel = 
        conv = 
        bias = 
        conv_output = 
    relu_output = 
    pool_output =
    with tf.variable_scope('fc') as scope: 
        fc_output = 

def cal_loss(image, label):

def train_model(loss):

def train():
    with tf.Graph().as_default():
        batch_size = 64
        image, label = train_input(batch_size)
        logits = build_model(image)
        loss = cal_loss(image, label)
        train_op = train_model(loss)
        
        class LoggerHook(tf.train.SessionRunHook):
            def begin(self):
            def before_run(self,):
            def after_run():

        with tf.train.MonitoredTrainingSession(hooks = [tf.train.StopAtStepHook(last_step = MAX_STEP), tf.train.NanTensorHook(loss), LoggerHook()]) as mon_sess:  
            while not mon_sess.should_stop():
                mon_sess.run(train_op)

if __name__ = '__main__':
    tf.app.run()
    train()
