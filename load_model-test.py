import keras
import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
# saver = tf.train.Saver()

# 先加载图和参数变量
saver = tf.train.import_meta_graph('.//model//tensorflow_model//tensorflow_4d_Model.meta')
model_path = tf.train.latest_checkpoint('.//model//tensorflow_model')
saver.restore(sess, model_path)

# 通过变量访问

x_image = sess.graph.get_tensor_by_name("input:0")
result = sess.graph.get_operation_by_name("output")
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")

images = np.ones(2 * 10 * 28 * 28 * 9 * 1)
images = np.reshape(images, [2, 10, 28, 28, 9, 1])
images = images.astype(np.float)


print(sess.run(result, feed_dict={ x_image: images, keep_prob: 1.0}))
# print(a)
# graph=tf.get_default_graph()
