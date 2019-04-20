import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.datasets.cifar10 import load_data

def next_batch(num, data, labels):
    idx = np.arange(0, len(data));
    np.random.shuffle(idx);
    idx = idx[:num];

    data_shuffle = [data[i] for i in idx];
    labels_shuffle = [labels[i] for i in idx];

    return np.asarray(data_shuffle), np.asarray(labels_shuffle);

def build_CNN_classifier(x):
    x_image = x;
    # 1. cnn layer (64 filter) 32 x 32 x 3
    W_conv1 = tf.Variable(tf.truncated_normal(shape = [5, 5, 3, 64], stddev = 5e-2));
    b_conv1 = tf.Variable(tf.constant(0.1, shape = [64]));
    h_conv1 = tf.nn.relu(tf.nn.conv2d(x_image, W_conv1, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv1);

    # 2. pooling layer 32 x 32 x 3 -> 16 x 16 x 3
    h_pool2 = tf.nn.max_pool(h_conv1, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME');

    # 3. cnn layer (64 filter) 16 x 16 x 3
    W_conv3 = tf.Variable(tf.truncated_normal(shape = [5, 5, 64, 128], stddev = 5e-2));
    b_conv3 = tf.Variable(tf.constant(0.1, shape = [128]));
    h_conv3 = tf.nn.relu(tf.nn.conv2d(h_pool2, W_conv3, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv3);

    # 4. polling layer 16 x 16 x 3 -> 8 x 8 x 3
    h_pool4 = tf.nn.max_pool(h_conv3, ksize = [1, 3, 3, 1], strides = [1, 2, 2, 1], padding = 'SAME');

    # 5. cnn layer (64filter) 8 x 8 x 3
    W_conv5 = tf.Variable(tf.truncated_normal(shape = [3, 3, 128, 256], stddev = 5e-2));
    b_conv5 = tf.Variable(tf.constant(0.1, shape = [256]));
    h_conv5 = tf.nn.relu(tf.nn.conv2d(h_pool4, W_conv5, strides = [1, 1, 1, 1], padding = 'SAME') + b_conv5);

    # 6. fully connected layer 64filter -> classify 1024 fitures
    W_fc6 = tf.Variable(tf.truncated_normal(shape = [8 * 8 * 64, 1024], stddev = 5e-2));
    b_fc6 = tf.Variable(tf.constant(0.1, shape = [1024]));

    # reshape images flat
    h_conv5_flat = tf.reshape(h_conv5, [-1, 8 * 8 * 64]);
    h_fc6 = tf.nn.relu(tf.matmul(h_conv5_flat, W_fc6) + b_fc6);

    # define dropout
    h_fc6_drop = tf.nn.dropout(h_fc6, keep_prob);

    # 7. fully connected layer 128 fitures -> 10 class
    W_fc7 = tf.Variable(tf.truncated_normal(shape = [1024, 10], stddev = 5e-2));
    b_fc7 = tf.Variable(tf.constant(0.1, shape = [1024]));
    logits = tf.matmul(h_fc6_drop, W_fc7) + b_fc7;
    y_pred = tf.nn.softmax(logits);

    return y_pred, logits;

# 인풋 아웃풋 데이터, 드롭아웃 확률을 입력받기위한 플레이스홀더를 정의합니다.
x = tf.placeholder(tf.float32, shape=[None, 32, 32, 3])
y = tf.placeholder(tf.float32, shape=[None, 10])
keep_prob = tf.placeholder(tf.float32)

# CIFAR-10 데이터를 다운로드하고 데이터를 불러옵니다.
(x_train, y_train), (x_test, y_test) = load_data()
# scalar 형태의 레이블(0~9)을 One-hot Encoding 형태로 변환합니다.
y_train_one_hot = tf.squeeze(tf.one_hot(y_train, 10),axis=1)
y_test_one_hot = tf.squeeze(tf.one_hot(y_test, 10),axis=1)

# Convolutional Neural Networks(CNN) 그래프를 생성합니다.
y_pred, logits = build_CNN_classifier(x)

# Cross Entropy를 비용함수(loss function)으로 정의하고, AdamOptimizer를 이용해서 비용 함수를 최소화합니다.
loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y, logits=logits))
train_step = tf.train.AdamOptimizer(0.001).minimize(loss)

# 정확도를 계산하는 연산을 추가합니다.
correct_prediction = tf.equal(tf.argmax(y_pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))

# 세션을 열어 실제 학습을 진행합니다.
with tf.Session() as sess:
  # 모든 변수들을 초기화한다.
  sess.run(tf.global_variables_initializer())

  # 10000 Step만큼 최적화를 수행합니다.
  for i in range(10000):
    batch = next_batch(100, x_train, y_train_one_hot.eval())

    # 100 Step마다 training 데이터셋에 대한 정확도와 loss를 출력합니다.
    if i % 100 == 0:
      train_accuracy = accuracy.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      loss_print = loss.eval(feed_dict={x: batch[0], y: batch[1], keep_prob: 1.0})
      print("Epoch : %d, accuracy : %f, loss : %f" % (i, train_accuracy, loss_print))

    # 20% 확률의 Dropout을 이용해서 학습을 진행합니다.
    sess.run(train_step, feed_dict = {x: batch[0], y: batch[1], keep_prob: 0.8})

  # 학습이 끝나면 테스트 데이터(10000개)에 대한 정확도를 출력합니다.
  test_accuracy = 0.0
  for i in range(10):
    test_batch = next_batch(1000, x_test, y_test_one_hot.eval())
    test_accuracy = test_accuracy + accuracy.eval(feed_dict={x: test_batch[0], y: test_batch[1], keep_prob: 1.0})
  test_accuracy = test_accuracy / 10;

  print("test accuracy: %f" % test_accuracy);
