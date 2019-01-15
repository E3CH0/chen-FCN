# from keras.models import load_model
# import keras
import numpy as np
# from data.minist_2d_seg_data_and_label import creat_minist_2d_seg_data
# import gc
from matplotlib import pyplot as plt

channel_data_file_path = ".//data//channel.npy"
channel_label_file_path = ".//data//river_3d_label.npy"
image_wide = 28
image_height = 28
image_depth = 10
wide_step = 1
height_step = 1
depth_step = 1
up = 51
down = 201
left = 0
right = 150
high = 0
low = 29
channel_data_list = np.load(channel_data_file_path)
channel_data_list = channel_data_list[high:low, 0:9, up:down, left:right]
channel_label_list = np.load(channel_label_file_path)
channel_label_list = channel_label_list[high:low, up:down, left:right]
channel_height = channel_data_list.shape[2]
channel_wide = channel_data_list.shape[3]
channel_depth = channel_data_list.shape[0]
sum = 20
flag = 0


file_path='.//data//test//test_channel_data_depth'
for d in range(0, channel_depth, depth_step):
    d_low = d + image_depth
    if d_low >= channel_depth:
        break

    train_channel_data = []
    # train_channel_label = []

    for c in range(0, channel_height, height_step):
        c_down = c + image_height
        if c_down >= channel_height:
            break

        for r in range(0, channel_wide, wide_step):
            r_right = r + image_wide
            if r_right >= channel_wide:
                break



# for c in range(0, channel_height, height_step):
#     c_down = c + image_height
#     if c_down >= channel_height:
#         break
#
#     for r in range(0, channel_wide, wide_step):
#         r_right = r + image_wide
#         if r_right >= channel_wide:
#             break
#
#         for d in range(0, channel_depth, depth_step):
#             d_low = d + image_depth
#             if d_low >= channel_depth:
#                 break

            train_image = channel_data_list[d:d_low, 0:9, c:c_down, r:r_right]
            # train_image_label = channel_label_list[d + image_depth // 2, c + image_height // 2, r + image_wide // 2]

            # if flag > 0 and train_image_label == 0:
            #     continue
            # if train_image_label == 1:
            #     flag -= 1
            # else:
            #     flag += 1

            train_channel_data.append(train_image)
            # train_channel_label.append(train_image_label)

            # train_image = channel_data_list[d:d_low, 0:9, c:c_down, r:r_right]
            # train_image_label = channel_label_list[d + image_depth // 2, c + image_height // 2, r + image_wide // 2]
            #
            # if flag > 0 and train_image_label == 0:
            #     continue
            # if train_image_label == 1:
            #     flag -= 1
            # else:
            #     flag += 1
            #
            # train_channel_data.append(train_image)
            # train_channel_label.append(train_image_label)

    np.save(file_path+str(d),np.array(train_channel_data))
# print(len(train_channel_data), len(train_channel_label))
