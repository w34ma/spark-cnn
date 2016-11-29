import tensorflow as tf
import numpy as np
import os
import glob

CIFAR10_PATH = '../cifar10-bin'

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

def train_input(batch_size):
    files = glob.glob(os.path.join(CIFAR10_PATH, "data_batch_*.bin"))
    print(files)
    files = tf.train.string_input_producer(files)
    input = read_data(files)
    return generate_batch(input.image, input.label, batch_size, min_queue_size)

