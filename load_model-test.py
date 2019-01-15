import numpy as np
import tensorflow as tf

sess = tf.InteractiveSession()
# saver = tf.train.Saver()

# 先加载图和参数变量
saver = tf.train.import_meta_graph('.//model//tensorflow_4d_Model.meta')
model_path= tf.train.latest_checkpoint('.//model')
saver.restore(sess,model_path)

# 通过变量访问

x_image = sess.graph.get_tensor_by_name("input:0")
images = [np.zeros(2 * 10 * 28 * 28 * 9 * 1)]
images = np.reshape(images, [2, 10, 28, 28, 9, 1])
images = images.astype(np.float)

result = sess.graph.get_operation_by_name("y_conv").outputs[0]
a = sess.run(result, feed_dict={x_image: images})
print(a)
# graph=tf.get_default_graph()
