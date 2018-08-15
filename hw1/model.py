import numpy as np
import tensorflow as tf

from utils import summary


def shallow(inputs, name, units, verbose=1):
    with tf.variable_scope(name):
        hidden = tf.layers.dense(inputs, units, activation=tf.nn.relu, name='hidden')
        outputs = tf.layers.dense(hidden, 1, name='outputs')
        
    if verbose: summary(name)
    return outputs


def medium(inputs, name, units, verbose=1):
    with tf.variable_scope(name):
        hidden = tf.layers.dense(inputs, units, activation=tf.nn.relu, name='hidden1')
        hidden = tf.layers.dense(hidden, units, activation=tf.nn.relu, name='hidden2')
        outputs = tf.layers.dense(hidden, 1, name='outputs')
        
    if verbose: summary(name)
    return outputs


def deep(inputs, name, units, verbose=1):
    with tf.variable_scope(name):
        hidden = tf.layers.dense(inputs, units, activation=tf.nn.relu, name='hidden1')
        hidden = tf.layers.dense(hidden, units, activation=tf.nn.relu, name='hidden2')
        hidden = tf.layers.dense(hidden, units, activation=tf.nn.relu, name='hidden3')
        hidden = tf.layers.dense(hidden, units, activation=tf.nn.relu, name='hidden4')
        outputs = tf.layers.dense(hidden, 1, name='outputs')
        
    if verbose: summary(name)
    return outputs


def cnn(inputs, name, units, verbose=1):
    with tf.variable_scope(name):
        inputs = tf.reshape(inputs, (-1, 28, 28, 1), name='inputs')
        conv1 = tf.layers.conv2d(inputs, units, (3, 3), padding='same', activation=tf.nn.relu, name='conv1')
        conv2 = tf.layers.conv2d(conv1 , units, (3, 3), padding='same', activation=tf.nn.relu, name='conv2')
        flatten = tf.reshape(conv2, (-1, np.prod(conv2.shape.as_list()[1:])), name='flatten')
        outputs = tf.layers.dense(flatten, 10, name='outputs')

    if verbose: summary(name)
    return outputs



if __name__ == '__main__':
    x = tf.placeholder(tf.float32, [None, 1])
    shallow(x, 'shallow', 288)
    medium(x, 'medium', 28)
    deep(x, 'deep', 16)
    
    x = tf.placeholder(tf.float32, [None, 784])
    cnn(x, 'cnn', 32)
