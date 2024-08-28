import numpy as np
from keras.layers import Input, Conv2D, MaxPooling2D, Conv2DTranspose, Dropout, Cropping2D, BatchNormalization, LeakyReLU
from keras.regularizers import l1, l2, l1_l2

class Layers(object):
    """A class providing different types of network layers

    Attributes:
    -----------
    num_bands: Number of available spectral bands
    num_cls: Number of classes
    patch_width: Width of image patches
    patch_height: Height of image patches
    activation: Activation function for intermediate conv layer
    final_activation: Activation function for final network layer
    kernel_regularizer: Regularizer function applied to the kernel weights matrix
    norm_momentum: Momentum for the weights' moving average
    dropout: Rate of dropped nodes in layer
    dropout_on_last_layer_only: Dropout layer just in last layer
    clip_pixels: Number of Pixels to be clipped horizontally/vertically
    """

    def __init__(self, params, num_bands, num_cls=2):
        self.num_bands = num_bands
        self.num_cls = num_cls

        self.patch_height = params['patch_height']
        self.patch_width = params['patch_width']

        self.activation = params['activation']
        self.final_activation = 'softmax'
        self.kernel_regularizer = None #l1_l2(l1=params['L1reg'], l2=params['L2reg'])
        self.norm_momentum = params['batch_norm_momentum']
        self.dropout_rate = params['dropout_rate']
        self.dropout_on_last_layer_only = params['dropout_on_last_layer_only']

    def input_layer(self):
        """Create the input layer of the network"""
        return Input((self.patch_width, self.patch_height, self.num_bands))

    def conv2d_layer(self, num_filters, input, name=''):
        """Create a 2D-convolution layer

        Parameter:
        num_filters: Number of output filters in the convolution
        input: Input Tensor from previous layer
        """
        if name != '':
            if self.activation == 'leaky_relu':
                conv_layer = Conv2D(num_filters, (3, 3), activation='linear', padding='same',
                    kernel_initializer='he_uniform', kernel_regularizer=self.kernel_regularizer, name=name)(input)
                return LeakyReLU(alpha=0.1, name='leaky_relu_' + name)(conv_layer)
            else:
                return Conv2D(num_filters, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_uniform', kernel_regularizer=self.kernel_regularizer, name=name)(input)
        else:
            if self.activation == 'leaky_relu':
                conv_layer = Conv2D(num_filters, (3, 3), activation='linear', padding='same',
                    kernel_initializer='he_uniform', kernel_regularizer=self.kernel_regularizer)(input)
                return LeakyReLU(alpha=0.1)(conv_layer)
            else:
                return Conv2D(num_filters, (3, 3), activation='relu', padding='same',
                    kernel_initializer='he_uniform', kernel_regularizer=self.kernel_regularizer)(input)

    def batch_norm_layer(self, input):
        """Create a batch normalization layer to regularize the data

        Parameter:
        input: Input Tensor from previous layer
        """
        return BatchNormalization(momentum=self.norm_momentum)(input)

    def pooling_layer(self, input):
        """Create a pooling layer to downsample the data

        Parameter:
        input: Input Tensor from previous layer
        """
        return MaxPooling2D(pool_size=(2, 2))(input)

    def upsampling_layer(self, num_filters, input):
        """Create an upsampling layer

        Parameter:
        num_filters: Number of output filters in the convolution
        input: Input Tensor from previous layer
        """
        return Conv2DTranspose(num_filters, kernel_size=(2, 2), strides=(2, 2), padding='same',
                               kernel_initializer='glorot_uniform')(input)

    def dropout_layer(self, input):
        """Create a dropout layer to decrease overfitting

        Parameter:
        input: Input Tensor from previous layer
        """
        return Dropout(self.dropout_rate)(input) if not self.dropout_on_last_layer_only else input

    def fully_con_layer(self, input):
        """Create the final layer of the network

        Parameter:
        input: Input Tensor from previous layer
        """
        return Conv2D(self.num_cls, (1, 1), activation=self.final_activation,
                      kernel_initializer='glorot_uniform', name='fully')(input)