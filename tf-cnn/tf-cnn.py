import tensorflow as tf
import numpy as np
import os
import glob
import time

CIFAR10_PATH = '../cifar10-bin'
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

def read_data(files):
    class cifar10(object):
        pass

    result = cifar10()
    result.label_size = LABEL_SIZE
    result.image_height = IMAGE_HEIGHT
    result.image_width = IMAGE_WIDTH
    result.image_depth = IMAGE_DEPTH
    result.image_size = result.image_height * result.image_width * result.image_depth
    result.entry_size = result.label_size + result.image_size
    
    reader = tf.FixedLengthRecordReader(record_bytes = result.entry_size)
    result.key, value = reader.read(files)
    entry = tf.decode_raw(value, tf.uint8)
    result.label = tf.cast(tf.slice(entry, [0], [result.label_size]), tf.int32)

    result.image = tf.reshape(tf.slice(entry, [result.label_size], [result.image_size]), [result.image_depth, result.image_height, result.image_width])

    result.image = tf.transpose(result.image, [1, 2, 0])
    result.image = tf.cast(result.image, tf.float32)

    return result

def generate_batch(image, label, batch_size, min_queue_size):
    thread_num = 16
    image, label = tf.train.shuffle_batch([image, label], batch_size = batch_size, num_threads = thread_num, capacity = min_queue_size + 3 * batch_size, min_after_dequeue = min_queue_size)
    return image, tf.one_hot(label, depth = LABEL_NUM)

def train_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "data_batch_*.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = np.int32(FRAC_MIN_QUEUE_SIZE * TRAIN_EPOCH_NUM) 
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

def test_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "test_batch.bin"))
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    min_queue_size = np.int32(FRAC_MIN_QUEUE_SIZE * TRAIN_EPOCH_NUM) 
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

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        image, label = train_input(BATCH_SIZE)
        logits = build_model(image)
        loss = cal_loss(logits, label)
        train_op = train_model(loss)
        
        class _LoggerHook(tf.train.SessionRunHook):
            def begin(self):
                self._step = -1;
        
            def before_run(self, run_context):
                self._step += 1
                self._start_time = time.time()
                return tf.train.SessionRunArgs(loss)
    
            def after_run(self, run_context, run_values):
                duration = time.time() - self._start_time
                loss_value = run_values.results
                if self._step % 100 == 0:
                    print("current loss value: %f" % loss_value)

        with tf.train.MonitoredTrainingSession(hooks=[tf.train.StopAtStepHook(last_step = ITERATION_NUM), tf.train.NanTensorHook(loss), _LoggerHook()]) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)     

def main(argv=None):
    train()

if __name__ == '__main__':
    tf.app.run()
