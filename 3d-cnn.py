import keras
from keras.datasets import mnist
from keras.layers import Conv3D, MaxPooling3D,Dropout
from keras.layers import Dense, Flatten, np
from keras.models import Sequential
import keras.applications.vgg16
# import numpy as np

batch_size = 128
num_classes = 10
epochs = 1

# input image dimensions
img_rows, img_cols, img_depths = 28, 28, 28

# the data, split between train and test sets
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)

# 转换成3d
x_train_3d = []
for train_index in range(x_train.shape[0]):
    x_train_3d.append([x_train[train_index] ]* img_depths)
x_train = np.array(x_train_3d)

x_test_3d = []
for test_index in range(x_test.shape[0]):
    x_test_3d.append([x_test[test_index]] * img_depths)
x_test = np.array(x_test_3d)

print(x_train.shape, x_test.shape)

# 格式
# x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depths, 1)
# x_test = x_test.reshape(x_test.shape[0], img_rows, img_cols, img_depths, 1)
x_train = x_train.reshape(x_train.shape[0], img_rows, img_cols, img_depths,1)
x_test = x_test.reshape(x_test.shape[0],img_rows, img_cols, img_depths, 1)
x_train = x_train[0:1280]
x_test = x_test[0:1280]
y_train = y_train[0:1280]
y_test = y_test[0:1280]

# 输入数据的格式
input_shape = (img_rows, img_cols, img_depths,1)

# 将像素值 规范在0-1
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255
print('x_train shape:', x_train.shape)
print('x_test shape:', x_test.shape)

# 将类别用类似二进制表示，例如类别1，2表示为10，01
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential()
model.add(Conv3D(32, kernel_size=(3, 3, 3),
                 activation='relu',
                 input_shape=input_shape))
model.add(Conv3D(64, (3, 3, 3), activation='relu'))
model.add(MaxPooling3D(pool_size=(2, 2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(num_classes, activation='softmax'))

# 编译，损失函数，优化器，
model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

# 训练
model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
            verbose=1)
# validation_data=(x_test, y_test))

# 测试
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])

