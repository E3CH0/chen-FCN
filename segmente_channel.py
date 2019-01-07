import keras

from data.train.load_channel_data_and_label import load_train_data_and_label
from model.cnn_3D_Model import cnn_3D

batch_size = 128
classes_num = 2
epochs = 1
img_depths, img_height, img_wide = 9, 33, 33

# load data
channel_train_data, channel_train_label = load_train_data_and_label()
channel_train_data = channel_train_data[:, 15, :, :, :]
channel_train_data = channel_train_data.reshape(channel_train_data.shape[0], img_depths, img_height, img_wide, 1)

channel_train_label = channel_train_label[:, 15]
channel_train_label = keras.utils.to_categorical(channel_train_label, classes_num)
print(channel_train_data.shape, channel_train_label.shape)

# input image dimensions
input_shape = (img_depths, img_height, img_wide, 1)

# build model
model = cnn_3D(input_shape=input_shape, classes_num=classes_num)

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(channel_train_data, channel_train_label,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1)

label_score = model.predict(channel_train_data, batch_size=batch_size)

image_point_sum = channel_train_label.shape[0]
predict_channel_point_nums = 0
for i in range(label_score.shape[0]):
    if label_score[i][0] < label_score[i][1]:
        predict_channel_point_nums += 1
predict_not_channel_point_nums = image_point_sum - predict_channel_point_nums

channel_point_nums = 0
for i in range(channel_train_label.shape[0]):
    if channel_train_label[i][0] < channel_train_label[i][1]:
        channel_point_nums += 1
not_channel_point_nums = image_point_sum - channel_point_nums


print("图像像素总数：" + str(image_point_sum), "河道像素总数：" + str(channel_point_nums),
      "非河道像素总数："+str(not_channel_point_nums), "预测河道像素点总数：" + str(predict_channel_point_nums)
      , "预测非河道像素点总数：" + str(predict_not_channel_point_nums))
