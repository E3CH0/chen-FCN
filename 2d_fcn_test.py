import gc

import keras
import numpy as np

from data.minist_2d_seg_data_and_label import creat_minist_2d_seg_data
from model.FCN32_VGG16_2D_Model import Fcn32
from model.cnn_2D_Model import cnn_2D_without_dense

num_classes = 11

batch_size = 1000
epochs = 100
num_trainData = 10000
num_testData = 100

# data
(x_train, y_train), (x_test, y_test) = creat_minist_2d_seg_data(num_trainData, num_testData)

# 0-1,float
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
x_train /= 255
x_test /= 255

# set type
x_train = x_train.reshape(x_train.shape[0], x_train.shape[1], x_train.shape[2], 1)
x_test = x_test.reshape(x_test.shape[0], x_test.shape[1], x_test.shape[2], 1)

y_train = np.reshape(y_train, newshape=(y_train.shape[0], -1))
y_test = np.reshape(y_test, newshape=(y_test.shape[0], -1))
print(y_train.shape)
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

input_shape = x_train.shape[1:]
print(input_shape)
print("x_train.shape, y_train.shape, x_test.shape, y_test.shape:", x_train.shape, y_train.shape, x_test.shape,
      y_test.shape)

gc.collect()
# model
front_model = cnn_2D_without_dense(input_shape, num_classes)
model = Fcn32(front_model, num_classes)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

history = model.fit(x_train, y_train,
                    batch_size=batch_size,
                    epochs=epochs,
                    verbose=2)

import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
handler = logging.FileHandler('.//Log//trainLog.log')
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
handler.setFormatter(formatter)
logger.addHandler(handler)

logger.info(history.history)
model.save('.//weight//fcn2d_model_weight.h5')
score = model.evaluate(x_test, y_test, verbose=0)
logger.info(score)
