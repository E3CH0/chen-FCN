from keras.models import load_model
import keras
import numpy as np
from data.minist_2d_seg_data_and_label import creat_minist_2d_seg_data
import gc
from matplotlib import pyplot as plt

num_classes = 11

batch_size = 1000
epochs = 100
num_trainData = 10000
num_testData = 10
model = load_model('.//weight//fcn2d_model_weight.h5')

(x_train, y_train), (x_test, y_test) = creat_minist_2d_seg_data(num_trainData, num_testData)

# 0-1,float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# set type
# x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

# y_train = np.reshape(y_train, newshape=(y_train.shape[0], -1))
# y_test = np.reshape(y_test, newshape=(y_test.shape[0], -1))
# print(y_train.shape)
# y_train = keras.utils.to_categorical(y_train, num_classes)
# y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]
print(input_shape)
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape:", x_train.shape, y_train.shape, x_test.shape,
      y_test.shape)

gc.collect()

y_test_predict=model.predict(x_test)
print(y_test_predict.shape)

# y_test_predict = y_test_predict.reshape(y_test_predict.shape[0], y_test.shape[2], y_test.shape[3])
print(y_test.shape)

# for i in range(y_test.shape[0]):
#       plt.subplot(2, 2, 1)
#       plt.imshow(y_test[i][0])
#       # plt.show()
#
#       plt.subplot(2, 2, 2)
#       plt.imshow(y_test_predict[i])
#       plt.show()

