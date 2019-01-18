import numpy as np
import tensorflow as tf
from matplotlib import pyplot as plt
import sys

sess = tf.InteractiveSession()
# saver = tf.train.Saver()

# 先加载图和参数变量
saver = tf.train.import_meta_graph('.//model//tensorflow_model//tensorflow_4d_Model.meta')
model_path = tf.train.latest_checkpoint('.//model//tensorflow_model')
saver.restore(sess, model_path)

# 通过变量访问

x_image = sess.graph.get_tensor_by_name("input:0")
keep_prob = sess.graph.get_tensor_by_name("keep_prob:0")
result = sess.graph.get_operation_by_name("output").outputs[0]
result_list = tf.argmax(result, 1)

images = np.ones(2 * 10 * 28 * 28 * 9 * 1)
images = np.reshape(images, [2, 10, 28, 28, 9, 1])
images = images.astype(np.float)

# ".//data//test//test_channel_data_depth8.npy,"
def test_and_save_result(file_path,save_path,test_batch=300):
    # data
    test_channel_data_depth = np.load(file_path)
    test_channel_data_depth = np.reshape(test_channel_data_depth, [-1, 10, 9, 28, 28, 1])
    test_channel_data_depth = np.transpose(test_channel_data_depth, (0, 1, 3, 4, 2, 5))
    print(test_channel_data_depth.shape)
    predict_result_list = []
    for i in range(0, test_channel_data_depth.shape[0], test_batch):
        feed_dict = {x_image: test_channel_data_depth[i:i + test_batch], keep_prob: 1.0}
        predict_result_list.extend(sess.run(result_list, feed_dict=feed_dict))
        print(len(predict_result_list))
    predict_result_list = np.array(predict_result_list)
    predict_result_list = np.reshape(predict_result_list, [122, 122])
    np.save(save_path, predict_result_list)

for i in range(0,19):
    file_parent_path=".//data//test"
    if len( sys.argv)>1:
        file_parent_path=sys.argv[1]
    file_path=file_parent_path+"//test_channel_data_depth"+str(i)+".npy"
    save_path=".//visual//data//test_channel_data_depth"+str(i)
    test_and_save_result(file_path,save_path)
