from keras.layers import Conv2DTranspose, Activation, Reshape
from keras.models import Model

from model.cnn_2D_Model import cnn_2D_without_dense


def Fcn32(model, nums_classes):
    front_output = model.output
    o = Conv2DTranspose(filters=nums_classes, kernel_size=(4, 4), strides=(4, 4), padding='valid',
                        activation=None, name='score')(front_output)
    o = Reshape((-1, nums_classes), name='reshape1')(o)
    o = Activation("softmax", name='activation1')(o)
    fcn32 = Model(inputs=model.input, outputs=o)

    return fcn32


if __name__ == '__main__':
    # import logging
    #
    # logger = logging.getLogger(__name__)
    # logger.setLevel(logging.INFO)
    # handler = logging.FileHandler('../Log/trainLog.log')
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s -\n %(message)s')
    # handler.setFormatter(formatter)
    # logger.addHandler(handler)

    input_shape = ( 28, 28,1)

    front_model = cnn_2D_without_dense(input_shape, 11)
    m = Fcn32(model=front_model, nums_classes=11)
    # logger.info(print(m.summary()))
    # logger.info(print(len(m.layers)))
    # print(len(m.layers))
