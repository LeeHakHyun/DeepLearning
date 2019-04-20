import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.examples.tutorials.mnist import input_data

mnist = input_data.read_data_sets("C:/tmp/data", one_hot=True)

X = tf.placeholder(tf.float32, [None, 28 * 28]);
Z = tf.placeholder(tf.float32, [None, 128]);

G_W1 = tf.Variable(tf.truncated_normal([128, 384], stddev = 1e-2));
G_b1 = tf.Variable(tf.zeros([384]));

G_W2 = tf.Variable(tf.truncated_normal([384, 28 * 28], stddev = 1e-2));
G_b2 = tf.Variable(tf.zeros([28 * 28]));

def generator(Z_noise):
    G_hidden = tf.nn.relu(tf.matmul(Z_noise, G_W1) + G_b1);
    G_output = tf.nn.sigmoid(tf.matmul(G_hidden, G_W2) + G_b2);
    return G_output;

D_W1 = tf.Variable(tf.truncated_normal([28 * 28, 384], stddev = 1e-2));
D_b1 = tf.Variable(tf.zeros([384]));

D_W2 = tf.Variable(tf.truncated_normal([384, 1], stddev = 1e-2));
D_b2 = tf.Variable(tf.zeros([1]));

def discriminator(X_input):
    D_hidden = tf.nn.relu(tf.matmul(X_input, D_W1) + D_b1);
    D_output = tf.nn.sigmoid(tf.matmul(D_hidden, D_W2) + D_b2);
    return D_output;

G = generator(Z);

loss_D = -tf.reduce_sum(tf.log(discriminator(X)) + tf.log(1 - discriminator(G)));
loss_G = -tf.reduce_sum(tf.log(discriminator(G)));

train_D = tf.train.AdamOptimizer(0.0002).minimize(loss_D, var_list=[D_W1, D_b1, D_W2, D_b2]);
train_G = tf.train.AdamOptimizer(0.0002).minimize(loss_G, var_list=[G_W1, G_b1, G_W2, G_b2]);

sess = tf.Session()
sess.run(tf.global_variables_initializer())

noise_test = np.random.normal(size=(10, 128)) # 10 = Test Sample Size, 128 = Noise Dimension

for epoch in range(200): # 200 = Num. of Epoch
    for i in range(int(mnist.train.num_examples / 100)): # 100 = Batch Size
        batch_xs, batch_ts = mnist.train.next_batch(100)
        noise = np.random.normal(size=(100, 128))

        sess.run(train_D, feed_dict={X: batch_xs, Z: noise})
        sess.run(train_G, feed_dict={Z: noise})

    if epoch == 0 or (epoch + 1) % 10 == 0: # 10 = Saving Period
        samples = sess.run(G, feed_dict={Z: noise_test})

        fig, ax = plt.subplots(1, 10, figsize=(10, 1))
        for i in range(10):
            ax[i].set_axis_off()
            ax[i].imshow(np.reshape(samples[i], (28, 28)))
        plt.savefig('samples_ex{}.png'.format(str(epoch).zfill(3)), bbox_inches='tight')
        print('samples_ex{}.png saved'.format(str(epoch).zfill(3)));
        plt.close(fig)
