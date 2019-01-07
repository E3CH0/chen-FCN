from keras.layers import Conv3DTranspose, Activation, Reshape
from keras.models import Model

from model.cnn_3D_Model import cnn_3D_without_dense


def Fcn32(model, nums_classes):
    front_output = model.output
    o = Conv3DTranspose(filters=nums_classes, kernel_size=(4, 4, 4), strides=(4, 4, 4), padding='valid',
                        activation=None, name='score')(front_output)
    o = Reshape((-1 ,nums_classes), name='reshape1')(o)
    o = Activation("softmax", name='activation1')(o)
    fcn32 = Model(inputs=model.input, outputs=o)

    return fcn32


if __name__ == '__main__':
    input_shape = (1, 28, 28, 28)

    front_model = cnn_3D_without_dense(input_shape, 11)
    m = Fcn32(model=front_model, nums_classes=11)
    m.summary()
    print(len(m.layers))
