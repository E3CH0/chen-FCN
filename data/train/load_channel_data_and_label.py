# # AzimuthValue:方位角，ChaosValue:混沌
# # CoherenceValue:相干，ContinuityValue:连续
# # DipValue:倾角，      FaultValue:断层
# # LineLikeValue:线条， PlaneLikeValue:平面
# attribute_list = ['amp', 'AzimuthValue', 'ChaosValue', 'CoherenceValue',
#                   'ContinuityValue', 'DipValue', 'FaultValue', 'LineLikeValue', 'PlaneLikeValue']
#
# # 加载属性
# file_path = "F://echo//bishe//code//DATA"
# channel_data_list = []
#
# for attribute in attribute_list:
#     attribute_path = file_path + "//" + attribute + ".raw"
#     attribute_list = np.fromfile(attribute_path, '<f4', -1)
#     attribute_list = np.reshape(attribute_list, [301, 201, 29])
#     attribute_list = np.transpose(attribute_list, (2, 1, 0))
#
#     channel_data_list.append(attribute_list)
#
# # 加载河道数据和标签，29 * 9 * 201 * 301
# channel_data_list = np.array(channel_data_list)
# print(channel_data_list.shape)
# channel_data_list = np.transpose(channel_data_list, (1, 0, 2, 3))
# print(channel_data_list.shape)
# np.save("F://echo//bishe//code//DATA//channel.npy", channel_data_list)

import numpy as np
import gc

# # AzimuthValue:方位角，ChaosValue:混沌
# # CoherenceValue:相干，ContinuityValue:连续
# # DipValue:倾角，      FaultValue:断层
# # LineLikeValue:线条， PlaneLikeValue:平面
# attribute_list = ['amp', 'AzimuthValue', 'ChaosValue', 'CoherenceValue',
#                   'ContinuityValue', 'DipValue', 'FaultValue', 'LineLikeValue', 'PlaneLikeValue']

#up down left right 是整个图片201*301的选择的部分位置区域
#image_wide/height 是训练数据图片的大小
#step是移动的距离
def load_train_data_and_label(channel_data_file_path="F://echo//bishe//code//DATA//channel.npy",
                              channel_label_file_path="F://echo//bishe//code//DATA//river_3d_label.npy",
                              image_wide=33, image_height=33, wide_step=5, height_step=5,
                              up=51, down=201, left=0, right=150):
    channel_data_list = np.load(channel_data_file_path)
    channel_data_list = channel_data_list[0:29, 0:9, up:down, left:right]

    channel_label_list = np.load(channel_label_file_path)
    channel_label_list = channel_label_list[0:29, up:down, left:right]

    train_channel_data = []
    train_channel_label = []
    channel_height = channel_data_list.shape[2]
    channel_wide = channel_data_list.shape[3]

    for c in range(0, channel_height, height_step):
        c_down = c + image_height
        if c_down >= channel_height:
            break
        for r in range(0, channel_wide, wide_step):
            r_right = r + image_wide
            if r_right >= channel_wide:
                break

            train_image = channel_data_list[0:29, 0:9, c:c_down, r:r_right]
            train_image_label = channel_label_list[0:29, c + image_height // 2, r + image_wide // 2]

            train_channel_data.append(train_image)
            train_channel_label.append(train_image_label)

    # test_channel_data = []
    # test_channel_label = []
    # for c in range(channel_height):
    #     c_down = c + image_height
    #     if c_down >= channel_height:
    #         break
    #     for r in range(channel_wide):
    #         r_right = r + image_wide
    #         if r_right >= channel_wide:
    #             break
    #
    #         train_image = channel_data_list[0:29, 0:9, c:c_down, r:r_right]
    #         train_image_label = channel_label_list[0:29, c + image_height // 2, r + image_wide // 2]
    #
    #         test_channel_data.append(train_image)
    #         test_channel_label.append(train_image_label)
    #

    #
    # # print(len(test_channel_data))
    # # print(len(test_channel_label))
    #
    # test_channel_data_part=[]
    # for i in range(0, 3000, 1000):
    #     test_channel_data_part.append(np.array(test_channel_data[i:i + 1000]))
    #     gc.collect()
    #
    #
    # test_channel_label = np.array(test_channel_label)

    train_channel_data = np.array(train_channel_data)
    train_channel_label = np.array(train_channel_label)
    return (train_channel_data, train_channel_label)


if __name__ == '__main__':
    (train_channel_data, train_channel_label) = load_train_data_and_label()
    print(train_channel_data.shape,train_channel_label.shape)
    # print(len(test_channel_data))
    # print("train_channel_data.shape:", train_channel_data.shape)
    # print("train_channel_label:", train_channel_label.shape)
