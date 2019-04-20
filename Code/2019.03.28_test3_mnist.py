import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data
np.random.seed(20190415);
tf.set_random_seed(20190415);
mnist = input_data.read_data_sets("C:/tmp/data/", one_hot = True);

def build_CNN_classfier(x):
    x_image = tf.reshape(x, [-1, 28, 28, 1]);

    # 1. cnn layer (64 filter) : 28 x 28 x 1 -> 28 x 28 x 64
    w_conv1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 1, 64], stddev = 5e-2));
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [64]));
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, w_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1);

    # 2. pooling layer : 28 x 28 x 64 -> 14 x 14 x 64
    h_pool2 = tf.nn.max_pool(h_conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME');

    # 3. cnn layer (128 filter) : 14 x 14 x 64 -> 14 x 14 x 128
    w_conv3 = tf.Variable(tf.truncated_normal(shape = [5, 5, 64, 128], stddev = 5e-2));
    b_conv3 = tf.Variable(tf.constant(0.1, shape = [128]));
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, w_conv3, strides = [1, 1, 1, 1] , padding = 'SAME') + b_conv3);

    # 4. pooling layer : 14 x 14 x 128 -> 7 x 7 x 128
    h_pool4 = tf.nn.max_pool(h_conv3, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME');

    #5 cnn layer (128 filter) : 7 x 7 x 128
    w_conv5 = tf.Variable(tf.truncated_normal(shape = [5, 5, 128, 128], stddev = 5e-2));
    b_conv5 = tf.Variable(tf.constant(0.1, shape = [128]));
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, w_conv5, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv5);

    # 6. fully conneted : 128 filter -> 1024 fitures
    w_fc6 = tf.Variable(tf.truncated_normal(shape = [7 * 7 * 128, 1024], stddev = 5e-2));
    b_fc6 = tf.Variable(tf.constant(0.1, shape = [1024]));

    # reshape images flat
    h_conv6_flat = tf.reshape(h_conv5, [-1, 7 * 7 * 128]);
    h_fc6 = tf.nn.relu(tf.matmul(h_conv6_flat, w_fc6) + b_fc6);

    # define dropout
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob);

    # 7. fully connected layer : 1024 fitures -> 10 class
    w_fc7 = tf.Variable(tf.truncated_normal(shape = [1024, 10], stddev = 5e-2));
    b_fc7 = tf.Variable(tf.constant(0.1, shape = [10]));
    logits = tf.matmul(h_fc6_drop, w_fc7) + b_fc7;
    y_pred = tf.nn.softmax(logits);

    return y_pred, logits;

x = tf.placeholder(tf.float32, [None, 784]);
p = tf.placeholder(tf.float32, [None, 10]);
keep_prob = tf.placeholder(tf.float32);

y_pred, logits = build_CNN_classfier(x);

loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels = p, logits = logits));
train_step = tf.train.AdamOptimizer(0.001).minimize(loss);

correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(p, 1));
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32));

sess = tf.InteractiveSession();
sess.run(tf.initialize_all_variables());

training_epochs = 15;
batch_size = 100;

for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)
    for i in range (total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        train_accuracy = accuracy.eval(feed_dict = {x: batch_xs, p: batch_ys, keep_prob: 1.0});
        loss_print = loss.eval(feed_dict = {x: batch_xs, p: batch_ys, keep_prob: 1.0});
        sess.run(train_step, feed_dict = {x: batch_xs, p: batch_ys, keep_prob: 0.8})

    print("Epoch : %d, accuracy : %f, loss : %f" %
            (epoch + 1, accuracy.eval(feed_dict = {x: mnist.test.images, p: mnist.test.labels, keep_prob : 1.0}), loss_print));
