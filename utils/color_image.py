def color_image(image, num_classes=20):
    import matplotlib as mpl
    import matplotlib.cm
    # 归一化
    norm = mpl.colors.Normalize(vmin=0., vmax=num_classes)
    # 得到colormap
    mycm = mpl.cm.get_cmap('Set1')
    return mycm(norm(image))


# test
# import matplotlib.image as mpimg  # mpimg 用于读取图片
#
# image_path = '..\\data\\train\images\\2007_000129.jpg'
# image = mpimg.imread(image_path)  # 读取和代码处于同一目录下的 lena.png
# print(image)
# color_map=color_image(image)
# print(color_map)
