import numpy as np
import scipy.misc

label_3d = []
# label_3d = np.array(label_3d)
for i in range(24):
    image_save_path = "F://echo//bishe//code//DATA//ampImage//label"
    image_save_path += "//river_depth_" + str(i) + "_json" + "//label.png"
    image = scipy.misc.imread(image_save_path)

    image = np.array(image)
    img_label = np.zeros((image.shape[0], image.shape[1]))
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            if image[i][j][0] > 1 or image[i][j][1] > 1 or image[i][j][2] > 1:
                img_label[i][j] = 1

    # print(img_label.shape)
    label_3d.append(img_label)

for i in range(24,29):
    img_label = np.zeros((201, 301))
    label_3d.append(img_label)


label_3d=np.array(label_3d)
print(label_3d.shape)
np.save("F://echo//bishe//code//DATA//river_3d_label.npy", label_3d)

a= np.load("F://echo//bishe//code//DATA//river_3d_label.npy")
print(a.shape)
print((a==label_3d).all())
#
# from matplotlib import pyplot as plt
#
# plt.imshow(img_label)
# plt.show()
