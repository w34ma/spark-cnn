from config import *

class cnn_model(object):

    def weight_variable(self, shape, name):
        with tf.device('/cpu:0'):
            initial = tf.truncated_normal_initializer(stddev = 0.1)
        return tf.get_variable(name, shape, initializer = initial)

    def bias_variable(self, shape, name):
        with tf.device('/cpu:0'):
            initial = tf.constant_initializer(0.1)
        return tf.get_variable(name, shape, initializer =  initial)

    def conv2d(self, x, W):
        return tf.nn.conv2d(x, W, strides = [1, 1, 1, 1], padding = 'SAME')

    def max_pool(self, x):
        return tf.nn.max_pool(x, ksize = [1, 2, 2, 1], strides = [1, 2, 2, 1], padding = 'SAME') 

    def init(self, x):

        with tf.variable_scope('conv1') as scope:
            conv_W = self.weight_variable([5, 5, 3, 32], 'weight')
            conv_b = self.bias_variable([32], 'bias')
        conv_out = self.conv2d(x, conv_W) + conv_b
        relu_out = tf.nn.relu(conv_out)
        pool_out = self.max_pool(relu_out)
        pool_out = tf.reshape(pool_out, [-1, 16 * 16 * 32])

        with tf.variable_scope('fc') as scope:
            fc_W = self.weight_variable([16 * 16 * 32, LABEL_NUM], 'weight')
            fc_b = self.bias_variable([LABEL_NUM], 'bias')
        fc_out = tf.matmul(pool_out, fc_W) + fc_b

        return fc_out

def cal_loss(cnn_y, y):
    loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(cnn_y, y))    
    return loss

def train_model(loss, global_step):
    train_step = tf.train.AdamOptimizer(1e-4).minimize(loss)
    variable_ave = tf.train.ExponentialMovingAverage(MOVING_AVE_DECAY, global_step)
    variable_ave_op = variable_ave.apply(tf.trainable_variables())
    with tf.control_dependencies([train_step, variable_ave_op]):
        train_res = tf.no_op(name = 'train')
    return train_step

def cal_acc(cnn_y, y):
    corr_pred = tf.equal(tf.argmax(cnn_y, 1), tf.argmax(y, 1))
    acc = tf.reduce_mean(tf.cast(corr_pred, tf.float32))
    return acc


