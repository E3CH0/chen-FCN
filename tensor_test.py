import keras
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


x_image = tf.placeholder("float", shape=[None, 10, 28, 28, 9, 1])

# W_conv1 = weight_variable([5, 5, 5, 5, 1, 8])
# b_conv1 = bias_variable([8])

layer1, layer1_weight, layer1_bias = new_conv_nd_layer(input=x_image, filter_size=[5, 5, 5, 5], num_filters=8,
                                                       pooling_type='max', pooling_strides=[1, 2, 2, 2, 1, 1],
                                                       pooling_ksize=[1, 2, 2, 2, 1, 1], pooling_padding='VALID',
                                                       strides=[1, 1, 1, 1, 1, 1], padding='SAME',
                                                       method='convolution')
print(layer1.shape.as_list())
# W_conv2 = weight_variable([5, 5, 5, 8, 16])
# b_conv2 = bias_variable([16])

layer2, layer2_weight, layer2_bias = new_conv_nd_layer(input=layer1, filter_size=[5, 5, 5, 5], num_filters=8,
                                                       pooling_type='max', pooling_strides=[1, 2, 2, 2, 1, 1],
                                                       pooling_ksize=[1, 2, 2, 2, 1, 1], pooling_padding='VALID',
                                                       strides=[1, 1, 1, 1, 1, 1], padding='SAME',
                                                       method='convolution')
print(layer2.shape.as_list())

W_fc1 = weight_variable([2 * 7 * 7 * 9 * 32, 512])
b_fc1 = bias_variable([512])

h_pool2_flat = tf.reshape(layer2, [-1, 2 * 7 * 7 * 9 * 32])
print(h_pool2_flat.shape.as_list())

h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)
print(h_fc1.shape.as_list())

keep_prob = tf.placeholder("float")
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)
print(h_fc1_drop.shape.as_list())

W_fc2 = weight_variable([512, 2])
b_fc2 = bias_variable([2])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
print(y_conv.shape.as_list())

y_ = tf.placeholder("float", [None, 2])

cross_entropy = -tf.reduce_sum(y_ * tf.log(y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)
correct_prediction = tf.equal(tf.argmax(y_conv, 1), tf.argmax(y_, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, "float"))
sess.run(tf.initialize_all_variables())

# images = [np.zeros(28 * 28 * 28 * 28), np.ones(28 * 28 * 28 * 28)]
# images=tf.reshape(images,[2,28,28,28,28,1])
# labels = [[1, 0], [0, 1]]

channel_data_file_path = ".//data//channel.npy"
channel_label_file_path = ".//data//river_3d_label.npy"
image_wide = 28
image_height = 28
image_depth = 10
wide_step = 5
height_step = 5
depth_step = 5
up = 51
down = 201
left = 0
right = 150
high = 0
low = 29

channel_data_list = np.load(channel_data_file_path)
channel_data_list = channel_data_list[high:low, 0:9, up:down, left:right]

channel_label_list = np.load(channel_label_file_path)
channel_label_list = channel_label_list[high:low, up:down, left:right]

channel_height = channel_data_list.shape[2]
channel_wide = channel_data_list.shape[3]
channel_depth = channel_data_list.shape[0]

for c in range(0, channel_height, height_step):
    c_down = c + image_height
    if c_down >= channel_height:
        break

    train_channel_data = []
    train_channel_label = []
    for r in range(0, channel_wide, wide_step):
        r_right = r + image_wide
        if r_right >= channel_wide:
            break

        for d in range(0, channel_depth, depth_step):
            d_low = d + image_depth
            if d_low >= channel_depth:
                break
            train_image = channel_data_list[d:d_low, 0:9, c:c_down, r:r_right]
            train_image_label = channel_label_list[d + image_depth // 2, c + image_height // 2, r + image_wide // 2]

            train_channel_data.append(train_image)
            train_channel_label.append(train_image_label)

        train_channel_data = np.array(train_channel_data)
        train_channel_label = np.array(train_channel_label)
        train_channel_data = np.reshape(train_channel_data, [-1, 10, 9, 28, 28, 1])
        train_channel_data = np.transpose(train_channel_data, (0, 1, 3, 4, 2, 5))
        print(train_channel_data.shape)

        train_channel_label = keras.utils.to_categorical(train_channel_label, 2)
        print(train_channel_label.shape)

        train_step.run(feed_dict={x_image: train_channel_data, y_: train_channel_label, keep_prob: 0.6})
    print("test accuracy %g" % accuracy.eval(feed_dict={
        x_image: train_channel_data, y_: train_channel_label, keep_prob: 1.0}))
# test_channel_data = []
# test_channel_label = []
# for c in range(channel_height):
#     c_down = c + image_height
#     if c_down >= channel_height:
#         break
#     for r in range(channel_wide):
#         r_right = r + image_wide
#         if r_right >= channel_wide:
#             break
#
#         train_image = channel_data_list[0:29, 0:9, c:c_down, r:r_right]
#         train_image_label = channel_label_list[0:29, c + image_height // 2, r + image_wide // 2]
#
#         test_channel_data.append(train_image)
#         test_channel_label.append(train_image_label)
#

#
# # print(len(test_channel_data))
# # print(len(test_channel_label))
#
# test_channel_data_part=[]
# for i in range(0, 3000, 1000):
#     test_channel_data_part.append(np.array(test_channel_data[i:i + 1000]))
#     gc.collect()
#
#
# test_channel_label = np.array(test_channel_label)





# print("test accuracy %g" % accuracy.eval(feed_dict={
#     x: images, y_: labels, keep_prob: 1.0}))
