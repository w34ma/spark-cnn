import tensorflow as tf
import numpy as np
import os
import glob
import time
from tf_pip_read import *
from tf_pip_model import *

CIFAR10_PATH = '../cifar10-bin'
CHECKPOINT_DIR = 'tf-backup'
TRAIN_EPOCH_NUM = 50000
TEST_EPOCH_NUM = 10000
ITERATION_NUM = 10000
LABEL_NUM = 10
DEVICE_REP = False

FRAC_MIN_QUEUE_SIZE = 0.4
BATCH_SIZE = 32
LABEL_SIZE = 1
IMAGE_HEIGHT = 32
IMAGE_WIDTH = 32
IMAGE_DEPTH = 3

def train():
    with tf.Graph().as_default():
        global_step = tf.contrib.framework.get_or_create_global_step()
        image, label = train_input(BATCH_SIZE)
        model = cnn_model()
        logits = model.init(image)
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

        with tf.train.MonitoredTrainingSession(hooks=[tf.train.StopAtStepHook(last_step = ITERATION_NUM), tf.train.NanTensorHook(loss), _LoggerHook()], config=tf.ConfigProto(log_device_placement = DEVICE_REP), checkpoint_dir = CHECKPOINT_DIR) as mon_sess:
            while not mon_sess.should_stop():
                mon_sess.run(train_op)     

def main(argv=None):
    if tf.gfile.Exists(CHECKPOINT_DIR):
        tf.gfile.DeleteRecursively(CHECKPOINT_DIR)
    tf.gfile.MakeDirs(CHECKPOINT_DIR)
    train()

if __name__ == '__main__':
    tf.app.run()
