import copy

import keras
from keras.datasets import mnist
from keras.layers import np


def transfer_3D_data(data, label, data_num):
    data_3d = []
    label_3d = []
    for index in range(data_num):
        label_2d = copy.copy(data[index])

        #seg label
        for r in range(label_2d.shape[0]):
            for c in range(label_2d.shape[1]):
                if label_2d[r][c] < 30:
                    label_2d[r][c] = 10
                else:
                    label_2d[r][c] = label[index]

        data_3d.append([data[index]] * 28)
        label_3d.append([label_2d] * 28)
    return np.array(data_3d), np.array(label_3d)


def creat_minist_3d_seg_data(train_num=60000, test_num=10000):
    # input image dimensions
    img_rows, img_cols, img_depths = 28, 28, 28

    # the data, split between train and test sets
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # 转换成3d
    (x_train, y_train) = transfer_3D_data(x_train, y_train, train_num)
    (x_test, y_test) = transfer_3D_data(x_test, y_test, test_num)

    return (x_train, y_train), (x_test, y_test)


if __name__ == '__main__':
    (x_train, y_train), (x_test, y_test) = creat_minist_3d_seg_data(100, 100)
    print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)
    y_train = keras.utils.to_categorical(y_train, 11)
    print(y_train.shape)
