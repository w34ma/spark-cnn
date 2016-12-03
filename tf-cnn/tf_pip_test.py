import tensorflow as tf
import numpy as np
import os
import glob
import time
import math
from tf_pip_read import *
from tf_pip_model import *

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

SLEEP_INTERVAL = 60

def test_one(saver, top_k):
    with tf.Session() as sess:
        ckpt = tf.train.get_checkpoint_state(CHECKPOINT_DIR)
        if ckpt and ckpt.model_checkpoint_path:
            saver.restore(sess, ckpt.model_checkpoint_path)
        else:
            print('No checkpoint file found')
            return
        coord = tf.train.Coordinator()
        try:
            threads = []
            for qr in tf.get_collection(tf.GraphKeys.QUEUE_RUNNERS):
                threads.extend(qr.create_threads(sess, coord = coord, daemon = True, start = True))
                
            num_iter = int(math.ceil(TEST_EPOCH_NUM / BATCH_SIZE))
            num_test = num_iter * BATCH_SIZE
            corr = 0
            step = 0
            while step < num_iter and not coord.should_stop():
                pred = sess.run(top_k)
                corr += np.sum(pred)
                step += 1
            print(corr)
            print(num_test)
            precision = 1.0 * corr / num_test
            print("accuracy: %f" % precision)

        except Exception as e:
            coord.request_stop(e) 
        coord.request_stop()
        coord.join(threads, stop_grace_period_secs = 10)

def test_all():
    with tf.Graph().as_default() as g:
        image, label = test_input(BATCH_SIZE)
        model = cnn_model()
        logits = model.init(image)
        top_k = tf.nn.in_top_k(logits, tf.argmax(label, axis = 1), 1)
        saver = tf.train.Saver()
    
        while True:
            print('new round test')
            test_one(saver, top_k)
            time.sleep(SLEEP_INTERVAL)

def main(argv=None):
    test_all()

if __name__ == '__main__':
    tf.app.run()
