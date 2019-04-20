#-*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data

(x_train, y_train), (x_test, y_test) = load_data();

batch_size = 100;
noise = 100;
total_epoch = 100;

X = tf.placeholder(tf.float32, [None, 32, 32, 3]);
Z = tf.placeholder(tf.float32, [None, noise]);
isTraining = tf.placeholder(tf.bool);

def generator(noise):
    with tf.variable_scope('Generator'):

        # reshape the noise
        output = tf.layers.dense(noise, 2 * 2 * 256);
        output = tf.reshape(output, [-1,  2, 2, 256]);
        output = tf.layers.batch_normalization(output, training = isTraining);
        output = tf.nn.leaky_relu(output, 0.2);
