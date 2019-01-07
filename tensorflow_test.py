import numpy as np
import tensorflow as tf

from model.conv4d.layers import new_conv_nd_layer

sess = tf.InteractiveSession()


def weight_variable(shape):
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)


def bias_variable(shape):
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)


def conv3d(x, W):
    return tf.nn.conv3d(x, W, strides=[1, 1, 1, 1, 1], padding='SAME')


def max_pool_2x2x2(x):
    return tf.nn.max_pool3d(x, ksize=[1, 2, 2, 2, 1],
                            strides=[1, 2, 2, 2, 1], padding='SAME')

x = tf.placeholder("float", shape=[None, 614656])

x_image = tf.reshape(x, [-1, 28, 28, 28, 28, 1])

# W_conv1 = weight_variable([5, 5, 5, 5, 1, 8])
# b_conv1 = bias_variable([8])

layer1, layer1_weight, layer1_bias = new_conv_nd_layer(input=x_image, filter_size=[5, 5, 5, 5], num_filters=8,
                                                       pooling_type='max', pooling_strides=[1, 2, 2, 2, 1, 1],
                                                       pooling_ksize=[1, 2, 2, 2, 1, 1], pooling_padding='VALID',
                                                       strides=[1, 1, 1, 1, 1, 1], padding='SAME',
                                                       method='convolution')

# W_conv2 = weight_variable([5, 5, 5, 8, 16])
# b_conv2 = bias_variable([16])

layer2, layer2_weight, layer2_bias = new_conv_nd_layer(input=x_image, filter_size=[5, 5, 5, 5], num_filters=16,
                                                       pooling_type='max', pooling_strides=[1, 2, 2, 2, 1, 1],
                                                       pooling_ksize=[1, 2, 2, 2, 1, 1], pooling_padding='VALID',
                                                       strides=[1, 1, 1, 1, 1, 1], padding='SAME',
                                                       method='convolution')


W_fc1 = weight_variable([7 * 7 * 7*7 * 16, 1024])
b_fc1 = bias_variable([1024])

h_pool2_flat = tf.reshape(layer2, [-1, 7 * 7 * 7*7 * 16])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)

y_ = tf.placeholder("float", [None, 2])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

images = [np.zeros(614656), np.ones(614656)]
# images=tf.reshape(images,[2,28,28,28,28,1])
labels = [[1, 0], [0, 1]]

for i in range(100):
    train_step.run(feed_dict={x: images, y_: labels, keep_prob: 1.0})

print("test accuracy %g" % accuracy.eval(feed_dict={
    x: images, y_: labels, keep_prob: 1.0}))
