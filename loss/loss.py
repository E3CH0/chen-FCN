# 损失函数，可以与任何优化器结合来使用优化整个模型

# future模块把下一个新版本的特性导入到当前版本，于是我们就可以在当前版本中测试一些新版本的特性
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf


def loss(logits, labels, num_classes, head=None):
    """计算模型和标签的损失率
    参数：
        logits：tensor，float -[batch_size,width,height,num_classes]

        labels: labels tensor,int32 -[batch_size,width,height,num_classes]

        head:numpy array -[num_classes]
            weighting the loss of each class
            optional:prioritize some classes
    :return
        loss：损失张量，float
    """

    with tf.name_scope('loss'):  # name_scope主要用来共享变量
        logits = tf.reshape(logits, (-1, num_classes))  # -1表示不用去指定这一维的大小
        labels=tf.to_float(tf.reshape(labels,(-1, num_classes)))

        epsilon = tf.constant(value=1e-4)#常数ε

        softmax=tf.nn.softmax(logits)+epsilon   #sofamax,归一化

        if head is not None:
            cross_entropy = -tf.reduce_sum(tf.multiply(labels * tf.log(softmax),
                                                       head), reduction_indices=[1])
        else:
            cross_entropy = -tf.reduce_sum(
                labels * tf.log(softmax), reduction_indices=[1])#压缩求和

        cross_entropy_mean = tf.reduce_mean(cross_entropy,
                                            name='xentropy_mean')

        tf.add_to_collection('losses', cross_entropy_mean)#KEY为name，value为list，这个方法作用，将变量添加到对应key的list中去

        loss=tf.add_n(tf.get_collection('losses'),name='total_loss')#列表元素相加

    return loss