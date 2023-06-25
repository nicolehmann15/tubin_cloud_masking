import numpy as np
from keras.models import Model
from keras.layers import concatenate
from src.architecture.layer import Layers

def u_net_model(params, num_bands, num_cls=2):
    """Create a UNet model considering specific model parameter

    Parameter:
    params: dict consisting of important model parameter
    num_bands: Number of available spectral bands
    num_cls: Number of classes
    """
    layer_manager = Layers(params, num_bands, num_cls)

    inputs = layer_manager.input_layer()
    # ---------------------------------------------------------------------
    # Encoder Network
    conv1 = layer_manager.conv2d_layer(32, inputs)
    conv1 = layer_manager.batch_norm_layer(conv1)
    conv1 = layer_manager.conv2d_layer(32, conv1)
    conv1 = layer_manager.batch_norm_layer(conv1)
    pool1 = layer_manager.pooling_layer(conv1)
    # ---------------------------------------------------------------------
    conv2 = layer_manager.conv2d_layer(64, pool1)
    conv2 = layer_manager.batch_norm_layer(conv2)
    conv2 = layer_manager.conv2d_layer(64, conv2)
    conv2 = layer_manager.batch_norm_layer(conv2)
    pool2 = layer_manager.pooling_layer(conv2)
    # ---------------------------------------------------------------------
    conv3 = layer_manager.conv2d_layer(128, pool2)
    conv3 = layer_manager.batch_norm_layer(conv3)
    conv3 = layer_manager.conv2d_layer(128, conv3)
    conv3 = layer_manager.batch_norm_layer(conv3)
    pool3 = layer_manager.pooling_layer(conv3)
    # ---------------------------------------------------------------------
    conv4 = layer_manager.conv2d_layer(256, pool3)
    conv4 = layer_manager.batch_norm_layer(conv4)
    conv4 = layer_manager.conv2d_layer(256, conv4)
    conv4 = layer_manager.batch_norm_layer(conv4)
    pool4 = layer_manager.pooling_layer(conv4)
    # ---------------------------------------------------------------------
    conv5 = layer_manager.conv2d_layer(512, pool4)
    conv5 = layer_manager.batch_norm_layer(conv5)
    conv5 = layer_manager.conv2d_layer(512, conv5)
    conv5 = layer_manager.batch_norm_layer(conv5)
    # ---------------------------------------------------------------------
    # Decoder Network
    up6 = concatenate([layer_manager.upsampling_layer(256, conv5), conv4])
    conv6 = layer_manager.conv2d_layer(256, up6)
    conv6 = layer_manager.dropout_layer(conv6)
    conv6 = layer_manager.conv2d_layer(256, conv6)
    conv6 = layer_manager.dropout_layer(conv6)
    # ---------------------------------------------------------------------
    up7 = concatenate([layer_manager.upsampling_layer(128, conv6), conv3])
    conv7 = layer_manager.conv2d_layer(128, up7)
    conv7 = layer_manager.dropout_layer(conv7)
    conv7 = layer_manager.conv2d_layer(128, conv7)
    conv7 = layer_manager.dropout_layer(conv7)
    # ---------------------------------------------------------------------
    up8 = concatenate([layer_manager.upsampling_layer(64, conv7), conv2])
    conv8 = layer_manager.conv2d_layer(64, up8)
    conv8 = layer_manager.dropout_layer(conv8)
    conv8 = layer_manager.conv2d_layer(64, conv8)
    conv8 = layer_manager.dropout_layer(conv8)
    # ---------------------------------------------------------------------
    up9 = concatenate([layer_manager.upsampling_layer(32, conv8), conv1])
    conv9 = layer_manager.conv2d_layer(32, up9)
    conv9 = layer_manager.dropout_layer(conv9)
    conv9 = layer_manager.conv2d_layer(32, conv9)
    conv9 = layer_manager.dropout_layer(conv9)
    # ---------------------------------------------------------------------
    # wofür cropping?
    crop9 = layer_manager.crop_layer(conv9)
    # ---------------------------------------------------------------------
    conv10 = layer_manager.fully_con_layer(conv9)
    # ---------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=conv10)
    return model