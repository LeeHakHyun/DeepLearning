import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

np.random.seed(20160612);
tf.set_random_seed(20160612);

mnist = input_data.read_data_sets("C:/tmp/data/", one_hot = True);

X = tf.placeholder(tf.float32, shape = [None, 28, 28]);
P = tf.placeholder(tf.float32, shape = [None, 10]);
keep_prob = tf.placeholder(tf.float32);

def CNN_classifier(x):
