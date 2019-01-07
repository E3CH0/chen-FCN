# from __future__ import absolute_import
# from __future__ import division
# from __future__ import print_function
#
# import logging
# import os
# import sys
#
# import numpy as np
# import tensorflow as tf
#
# VGG_MEAN = [103.939, 116.779, 123.68]
#
#
# class FCN8_VGG:
#     # 初始化加载vgg16.npy文件
#     def __init__(self, vgg16_npy_path):
#         if not os.path.isfile(vgg16_npy_path):
#             logging.error(("File '%s' not found. Download it from "
#                            "ftp://mi.eng.cam.ac.uk/pub/mttt2/"
#                            "models/vgg16.npy"), vgg16_npy_path)
#             sys.exit(1)
#         self.data_dict = np.load(vgg16_npy_path, encoding='latinl').item()  # item用于转换，将ndrarry转换成dict
#         self.wd = 5e-4  # todo ???
#         print("npy file loaded")
#
#     def build(self, rgb, train=False, num_classes=20, random_init_fc8=False,
#               debug=False, use_dilated=False):
#         """
#         使用加载好的权重构建VGG
#         Parameters
#         ----------
#         rgb: image batch tensor 图像映射到[0,255]
#             Image in rgb shap. Scaled to Intervall [0, 255]
#
#         train: bool 建立推理图还是训练图
#
#         num_classes: int 预测的种类数目
#
#         random_init_fc8 : bool 是否随机初始化fc8层，在这种情况下需要使用微调。
#
#         debug: bool 是否打印调试信息
#         """
#
#         with tf.name_scope('Processing'):
#             red, green, blue = tf.split(rgb, 3, 3)
#
#             bgr = tf.concat([
#                 blue - VGG_MEAN[0],
#                 green - VGG_MEAN[1],
#                 red - VGG_MEAN[2],
#             ], 3)  # concat ,表示将第三维相连，a*b*c与a*b*c变成a*b*2c
#
#             # input 打印变量的名字，data，为list，里面包含打印的内容，message为需要输出的错误信息
#             # first_n指只记录前n次，summarize对每个tensor只打印的条目数量，name是op的名字
#             # tf.Print()只是构建一个op，需要run之后才会打印
#             if debug:
#                 bgr = tf.Print(bgr, [tf.shape(bgr)],
#                                message='Shape of input image: ',
#                                summarize=4, first_n=1)
#
#             self.conv1_1 = self._conv_layer(bgr, "conv1_1")
#             self.conv1_2 = self._conv_layer(self.conv1_1, "conv1_2")
#
#     def _conv_layer(self, bottom, name):
#         with tf.variable_scope(name) as scope:
#             filt = self.get_conv_filter(name)
#             #tf.nn.conv2d(input, filter, strides, padding, use_cudnn_on_gpu=None, name=None)
#             #第一个参数input：指需要做卷积的输入图像，它要求是一个Tensor，
#             # 具有[batch, in_height, in_width, in_channels]这样的shape
#             #第二个参数filter：相当于CNN中的卷积核，它要求是一个Tensor，
#             # 具有[filter_height, filter_width, in_channels, out_channels]这样的shape
#             conv = tf.nn.conv2d(bottom, filt, [1, 1, 1, 1], padding='SAME')
#
#             conv_biases = self.get_bias(name)
#             bias = tf.nn.bias_add(conv, conv_biases)
#
#             relu = tf.nn.relu(bias)
#             # Add summary to Tensorboard
#             _activation_summary(relu)
#             return relu
#
#     # 从vgg中获取卷积核参数
#     def get_conv_filter(self, name):
#         #将变量初始化为给定常量
#         init = tf.constant_initializer(value=self.data_dict[name][0], dtype=tf.float32)
#         shape=self.data_dict[name][0].shape
#
#         print('Layer name: %s' % name)
#         print('Layer shape: %s' % str(shape))
#
#         #get_variable 共享变量
#         var = tf.get_variable(name="filter", initializer=init, shape=shape)
#         if not tf.get_variable_scope().reuse:
#             weight_decay = tf.multiply(tf.nn.l2_loss(var), self.wd,
#                                        name='weight_loss')#L2范数不但可以防止过拟合，还可以让我们的优化求解变得稳定和快速。
#             tf.add_to_collection(tf.GraphKeys.REGULARIZATION_LOSSES,
#                                  weight_decay)
#         _variable_summaries(var)
#         return var
#
#     def get_bias(self, name, num_classes=None):
#         bias_wights = self.data_dict[name][1]
#         shape = self.data_dict[name][1].shape
#         if name == 'fc8':
#             bias_wights = self._bias_reshape(bias_wights, shape[0],
#                                              num_classes)
#             shape = [num_classes]
#         init = tf.constant_initializer(value=bias_wights,
#                                        dtype=tf.float32)
#         var = tf.get_variable(name="biases", initializer=init, shape=shape)
#         _variable_summaries(var)
#         return var
#
#     def _bias_reshape(self, bweight, num_orig, num_new):
#         """ Build bias weights for filter produces with `_summary_reshape`
#
#         """
#         n_averaged_elements = num_orig // num_new
#         avg_bweight = np.zeros(num_new)
#         for i in range(0, num_orig, n_averaged_elements):
#             start_idx = i
#             end_idx = start_idx + n_averaged_elements
#             avg_idx = start_idx // n_averaged_elements
#             if avg_idx == num_new:
#                 break
#             avg_bweight[avg_idx] = np.mean(bweight[start_idx:end_idx])
#         return avg_bweight
#
#     def _bias_variable(self, shape, constant=0.0):
#         initializer = tf.constant_initializer(constant)
#         var = tf.get_variable(name='biases', shape=shape,
#                               initializer=initializer)
#         _variable_summaries(var)
#         return var
#
# def _activation_summary(x):
#     """Helper to create summaries for activations.
#
#     Creates a summary that provides a histogram of activations.
#     Creates a summary that measure the sparsity of activations.
#
#     Args:
#       x: Tensor
#     Returns:
#       nothing
#     """
#     # Remove 'tower_[0-9]/' from the name in case this is a multi-GPU training
#     # session. This helps the clarity of presentation on tensorboard.
#     tensor_name = x.op.name
#     # tensor_name = re.sub('%s_[0-9]*/' % TOWER_NAME, '', x.op.name)
#     tf.summary.histogram(tensor_name + '/activations', x)
#     tf.summary.scalar(tensor_name + '/sparsity', tf.nn.zero_fraction(x))
#
#
# def _variable_summaries(var):
#     """Attach a lot of summaries to a Tensor."""
#     if not tf.get_variable_scope().reuse:
#         name = var.op.name
#         logging.info("Creating Summary for: %s" % name)
#         with tf.name_scope('summaries'):
#             mean = tf.reduce_mean(var)
#             tf.summary.scalar(name + '/mean', mean)
#             with tf.name_scope('stddev'):
#                 stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#
#             # 1、tf.summary.scalar
#             # 用来显示标量信息，其格式为：
#             # tf.summary.scalar(tags, values, collections=None, name=None)
#             # 例如：tf.summary.scalar('mean', mean)
#             # 一般在画loss, accuary时会用到这个函数。
#             tf.summary.scalar(name + '/sttdev', stddev)
#             tf.summary.scalar(name + '/max', tf.reduce_max(var))
#             tf.summary.scalar(name + '/min', tf.reduce_min(var))
#
#             # 2、tf.summary.histogram
#             # 用来显示直方图信息，其格式为：
#             # tf.summary.histogram(tags, values, collections=None, name=None)
#             tf.summary.histogram(name, var)