import tensorflow as tf

x_data = [1,2,3]
y_data = [3,5,7]

W = tf.Variable(tf.random_uniform([1], -5.0, 5.0))
b = tf.Variable(tf.random_uniform([1], -3.0, 3.0))

X = tf.placeholder(tf.float32)
Y = tf.placeholder(tf.float32)
hypothesis = W * X + b
cost = tf.reduce_mean(tf.square(hypothesis - Y))
# Minimize
a = tf.Variable(0.1) # Learning rate, alpha
optimizer = tf.train.GradientDescentOptimizer(a)
train = optimizer.minimize(cost)

init = tf.initialize_all_variables()
sess = tf.Session()
sess.run(init)

for step in range(301):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step % 20 == 0:
        print (step, sess.run(cost, feed_dict={X:x_data, Y:y_data}), sess.run(W), sess.run(b))

print ('if X = 5 : ',sess.run(hypothesis, feed_dict={X : 5}))
print ('if X = 7 : ',sess.run(hypothesis, feed_dict={X : 7}))
