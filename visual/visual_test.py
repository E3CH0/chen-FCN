from matplotlib import pyplot as plt

from data.minist_3d_seg_data_and_label import creat_minist_3d_seg_data
from mpl_toolkits.mplot3d import Axes3D

# (x_train, y_train), (x_test, y_test) = mnist.load_data()
(x_train, y_train), (x_test, y_test) = creat_minist_3d_seg_data(100, 100)
# (100, 28, 28, 28) (100, 28, 28, 28) (100, 28, 28, 28) (100, 28, 28, 28)
x_picture = x_train[0][0]
y_picture = y_train[0][0]

# x_3d_pitcure = x_train[0]

# fig = plt.figure()
# ax = Axes3D(fig)

# ax = plt.subplot(111, projection='3d')  # 创建一个三维的绘图工程
# for i in range(x_3d_pitcure.shape[0]):
#     for j in range(x_3d_pitcure.shape[1]):
#         for k in range(x_3d_pitcure.shape[2]):
#             if x_3d_pitcure[i][j][k] > 0:
#                 ax.scatter(i, j, k, c='y')  # 绘制数据点
#
# ax.set_zlabel('Z')  # 坐标轴
# ax.set_ylabel('Y')
# ax.set_xlabel('X')
# plt.show()


plt.imshow(x_picture, cmap='gray')
plt.show()

plt.imshow(y_picture, cmap='gray')
plt.show()
