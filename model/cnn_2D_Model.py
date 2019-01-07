from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Input,Deconvolution3D,Conv3DTranspose
from keras.models import Model


# input_shape-[channel,row,col,depth]
def cnn_2D_without_dense(input_shape, nums_classes):
    input_layer = Input(input_shape)

    # Block1
    p1 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same', input_shape=input_shape,
                name='block1_conv1')(input_layer)
    p1 = Conv2D(8, kernel_size=(3, 3), activation='relu', padding='same',
                name='block1_conv2')(p1)
    p1 = MaxPooling2D((2, 2), name='block1_pool')(p1)

    # Block2
    p1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
                name='block2_conv1')(p1)
    p1 = Conv2D(16, kernel_size=(3, 3), activation='relu', padding='same',
                name='block2_conv2')(p1)
    p1 = MaxPooling2D((2, 2), name='block2_pool')(p1)

    # # Block3
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block3_conv1')(p1)
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block3_conv2')(p1)
    # p1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block3_pool')(p1)

    # # Block4
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block4_conv1')(p1)
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block4_conv2')(p1)
    # p1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block4_pool')(p1)
    #
    # # Block5
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block5_conv1')(p1)
    # p1 = Conv3D(32, kernel_size=(3, 3, 3), activation='relu', padding='same',
    #             name='block5_conv2')(p1)
    # p1 = MaxPooling3D((2, 2, 2), strides=(2, 2, 2), name='block5_pool')(p1)

    # fc
    # p1 = Flatten()(p1)
    # p1 = Dense(512, activation='relu', name='fc1')(p1)
    # p1 = Dense(512, activation='relu', name='fc2')(p1)
    # p1 = Dense(classes_num, activation='softmax', name='predictions')(p1)

    return Model(inputs=input_layer, outputs=p1)

if __name__ == '__main__':
    m=cnn_2D_without_dense(input_shape=(1, 28, 28), nums_classes=11)
    m.summary()
    print(len(m.layers))