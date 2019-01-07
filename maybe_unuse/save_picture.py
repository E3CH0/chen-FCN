import numpy as np
from matplotlib import pyplot as plt
import pylab
import scipy.misc


#将每个深度的图片保存





# read  image
image_path = "F://echo//bishe//code//DATA//amp.raw"
river_3d_list = np.fromfile(image_path, '<f4', -1)
river_3d_list = np.reshape(river_3d_list, [301, 201, 29])
river_3d_list = np.transpose(river_3d_list, (2, 1, 0))
# river_3d_list = river_3d_list.astype(int)
# print(river_3d_list.dtype)
# 29*201*301 宽 301 高 201 深 29

# 储存每个深度的河道图片
for i in range(river_3d_list.shape[0]):
    image = river_3d_list[i]
    min = np.min(image)
    max=np.max(image)


    image_save_path = "F://echo//bishe//code//DATA//ampImage"
    image_save_path += "//river_depth_" + str(i) + ".bmp"
    scipy.misc.toimage(image, cmin=min, cmax=max).save(image_save_path)

    # plt.imshow(image, cmap="magma")  # hsv(2) magma(3)
    # # plt.axis('off')  # 不显示坐标轴
    # plt.axes().get_xaxis().set_visible(False)
    # plt.axes().get_yaxis().set_visible(False)
    # plt.savefig(image_save_path,bbox_inches='tight')
    # plt.show()


