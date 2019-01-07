import copy

import keras
from keras.datasets import mnist
from keras.layers import np


def transfer_2D_data(data, label, data_num):
    label_2d = []
    for index in range(data_num):
        cur_label_2d = copy.copy(data[index])

        #seg label
        for r in range(cur_label_2d.shape[0]):
            for c in range(cur_label_2d.shape[1]):
                if cur_label_2d[r][c] < 30:
                    cur_label_2d[r][c] = 10
                else:
                    cur_label_2d[r][c] = label[index]

        label_2d.append([cur_label_2d])

    return np.array(data[0:data_num]), np.array(label_2d)


def creat_minist_2d_seg_data(train_num=60000, test_num=10000):
    # input image dimensions
    img_rows, img_cols = 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 转换成2d
    (x_train, y_train) = transfer_2D_data(x_train, y_train, train_num)
    (x_test, y_test) = transfer_2D_data(x_test, y_test, test_num)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = creat_minist_2d_seg_data(10, 10)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = keras.utils.to_categorical(y_train, 11)
    print(y_train.shape)
