from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

import matplotlib
import matplotlib.pyplot as plt

FILTERS_1 = 200
FILTERS_2 = 400
HIDDEN_N = 1024
NUM_ITERS = 2000
BATCH_SIZE = 100
DROPOUT_RATE = 0.5

x = tf.placeholder(tf.float32, shape=[None, 784])
y_ = tf.placeholder(tf.float32, shape=[None, 10])

"""
ex_image = mnist.test.images[0,:].reshape((28, 28))
plt.figure()
plt.imshow(ex_image)
plt.show()
"""

def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)
    
def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)
    
def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME')
    
def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')
    
W_conv1 = weight_variable([5, 5, 1, FILTERS_1])
b_conv1 = bias_variable([FILTERS_1])

x_image = tf.reshape(x, [-1, 28, 28, 1])
h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1)

W_conv2 = weight_variable([5, 5, FILTERS_1, FILTERS_2])
b_conv2 = bias_variable([FILTERS_2])
h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2)

W_fc1 = weight_variable([7 * 7 * FILTERS_2, HIDDEN_N])
b_fc1 = bias_variable([HIDDEN_N])
h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*FILTERS_2])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder(tf.float32)
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([HIDDEN_N, 10])
b_fc2 = bias_variable([10])

y_conv = tf.matmul(h_fc1_drop, W_fc2) + b_fc2

cross_entropy = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=y_, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
sess.run(tf.global_variables_initializer())
for i in range(NUM_ITERS):
    batch = mnist.train.next_batch(BATCH_SIZE)
    if (i % 10 == 0):
        train_accuracy = accuracy.eval(feed_dict={
            x:batch[0], y_:batch[1], keep_prob: 1.0})
        print("step %d, training accuracy %g" % (i, train_accuracy))
    train_step.run(feed_dict={x: batch[0], y_: batch[1], keep_prob: DROPOUT_RATE})

print("test accuracy %g" % accuracy.eval(feed_dict={x: mnist.test.images[:500, :], y_: mnist.test.labels[:500, :], keep_prob: 1.0}))



















    
