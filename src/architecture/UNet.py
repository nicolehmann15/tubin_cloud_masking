import tensorflow as tf
import numpy as np
from keras.applications import ResNet50
from keras.models import Model
from keras.layers import concatenate
from .layer import Layers

def u_net_model(params, num_bands, num_cls=2):
    """Create a UNet model considering specific model parameter

    Parameter:
    params: dict consisting of important model parameter
    num_bands: Number of available spectral bands
    num_cls: Number of classes
    """
    layer_manager = Layers(params, num_bands, num_cls)
    feature_size = params["starting_feature_size"]
    inputs = layer_manager.input_layer()
    # ---------------------------------------------------------------------
    # Encoder Network
    conv1 = layer_manager.conv2d_layer(feature_size, inputs)
    conv1 = layer_manager.batch_norm_layer(conv1)
    conv1 = layer_manager.conv2d_layer(feature_size, conv1)
    conv1 = layer_manager.batch_norm_layer(conv1)
    pool1 = layer_manager.pooling_layer(conv1)
    # ---------------------------------------------------------------------
    conv2 = layer_manager.conv2d_layer(feature_size * 2, pool1)
    conv2 = layer_manager.batch_norm_layer(conv2)
    conv2 = layer_manager.conv2d_layer(feature_size * 2, conv2)
    conv2 = layer_manager.batch_norm_layer(conv2)
    pool2 = layer_manager.pooling_layer(conv2)
    # ---------------------------------------------------------------------
    conv3 = layer_manager.conv2d_layer(feature_size * 4, pool2)
    conv3 = layer_manager.batch_norm_layer(conv3)
    conv3 = layer_manager.conv2d_layer(feature_size * 4, conv3)
    conv3 = layer_manager.batch_norm_layer(conv3)
    pool3 = layer_manager.pooling_layer(conv3)
    # ---------------------------------------------------------------------
    conv4 = layer_manager.conv2d_layer(feature_size * 8, pool3)
    conv4 = layer_manager.batch_norm_layer(conv4)
    conv4 = layer_manager.conv2d_layer(feature_size * 8, conv4)
    conv4 = layer_manager.batch_norm_layer(conv4)
    pool4 = layer_manager.pooling_layer(conv4)
    # ---------------------------------------------------------------------
    conv5 = layer_manager.conv2d_layer(feature_size * 16, pool4)
    conv5 = layer_manager.batch_norm_layer(conv5)
    conv5 = layer_manager.conv2d_layer(feature_size * 16, conv5)
    conv5 = layer_manager.batch_norm_layer(conv5)
    # ---------------------------------------------------------------------
    # Decoder Network
    up6 = concatenate([layer_manager.upsampling_layer(feature_size * 8, conv5), conv4])
    conv6 = layer_manager.conv2d_layer(feature_size * 8, up6)
    conv6 = layer_manager.dropout_layer(conv6)
    conv6 = layer_manager.conv2d_layer(feature_size * 8, conv6)
    conv6 = layer_manager.dropout_layer(conv6)
    # ---------------------------------------------------------------------
    up7 = concatenate([layer_manager.upsampling_layer(feature_size * 4, conv6), conv3])
    conv7 = layer_manager.conv2d_layer(feature_size * 4, up7)
    conv7 = layer_manager.dropout_layer(conv7)
    conv7 = layer_manager.conv2d_layer(feature_size * 4, conv7)
    conv7 = layer_manager.dropout_layer(conv7)
    # ---------------------------------------------------------------------
    up8 = concatenate([layer_manager.upsampling_layer(feature_size * 2, conv7), conv2])
    conv8 = layer_manager.conv2d_layer(feature_size * 2, up8)
    conv8 = layer_manager.dropout_layer(conv8)
    conv8 = layer_manager.conv2d_layer(feature_size * 2, conv8)
    conv8 = layer_manager.dropout_layer(conv8)
    # ---------------------------------------------------------------------
    up9 = concatenate([layer_manager.upsampling_layer(feature_size, conv8), conv1])
    conv9 = layer_manager.conv2d_layer(feature_size, up9)
    conv9 = layer_manager.dropout_layer(conv9)
    conv9 = layer_manager.conv2d_layer(feature_size, conv9)
    conv9 = layer_manager.dropout_layer(conv9)
    # ---------------------------------------------------------------------
    conv10 = layer_manager.fully_con_layer(conv9)
    # ---------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=conv10)
    return model

def transfer_l8_model(unet_instance, params, num_bands, num_cls=2):
    """Create a UNet model considering specific model parameter

    Parameter:
    params: dict consisting of important model parameter
    num_bands: Number of available spectral bands
    num_cls: Number of classes
    """
    layer_manager = Layers(params, num_bands, num_cls)
    feature_size = params["starting_feature_size"]

    unet_instance.load_model('./../models/strongest-weights-Landsat8_256_pp_md_BIG04_05_21.hdf5', '', 'mIoU_loss', False)
    # ---------------------------------------------------------------------
    # Transfered L8 Encoder Network
    # ---------------------------------------------------------------------
    input_layer = unet_instance.model.get_layer("input_1").output
    # 1st Block
    conv2d_layer = unet_instance.model.get_layer("conv2d")
    conv2d_layer.trainable = False  # makes layer/weights untrainable
    batch_norm_layer = unet_instance.model.get_layer("batch_normalization")
    batch_norm_layer.trainable = False
    conv2d_1_layer = unet_instance.model.get_layer("conv2d_1")
    conv2d_1_layer.trainable = False
    batch_norm_1_layer = unet_instance.model.get_layer("batch_normalization_1")
    batch_norm_1_layer.trainable = False
    block1 = unet_instance.model.get_layer("batch_normalization_1").output
    # ---------------------------------------------------------------------
    # 2nd Block
    conv2d_2_layer = unet_instance.model.get_layer("conv2d_2")
    conv2d_2_layer.trainable = False
    batch_norm_2_layer = unet_instance.model.get_layer("batch_normalization_2")
    batch_norm_2_layer.trainable = False
    conv2d_3_layer = unet_instance.model.get_layer("conv2d_3")
    conv2d_3_layer.trainable = False
    batch_norm_3_layer = unet_instance.model.get_layer("batch_normalization_3")
    batch_norm_3_layer.trainable = False
    block2 = unet_instance.model.get_layer("batch_normalization_3").output
    # ---------------------------------------------------------------------
    # 3rd Block
    conv2d_4_layer = unet_instance.model.get_layer("conv2d_4")
    conv2d_4_layer.trainable = False
    batch_norm_4_layer = unet_instance.model.get_layer("batch_normalization_4")
    batch_norm_4_layer.trainable = False
    conv2d_5_layer = unet_instance.model.get_layer("conv2d_5")
    conv2d_5_layer.trainable = False
    batch_norm_5_layer = unet_instance.model.get_layer("batch_normalization_5")
    batch_norm_5_layer.trainable = False
    block3 = unet_instance.model.get_layer("batch_normalization_5").output
    # ---------------------------------------------------------------------
    # 4th Block
    conv2d_6_layer = unet_instance.model.get_layer("conv2d_6")
    conv2d_6_layer.trainable = False
    batch_norm_6_layer = unet_instance.model.get_layer("batch_normalization_6")
    batch_norm_6_layer.trainable = False
    conv2d_7_layer = unet_instance.model.get_layer("conv2d_7")
    conv2d_7_layer.trainable = False
    batch_norm_7_layer = unet_instance.model.get_layer("batch_normalization_7")
    batch_norm_7_layer.trainable = False
    block4 = unet_instance.model.get_layer("batch_normalization_7").output
    # ---------------------------------------------------------------------
    # 5th Block
    conv2d_8_layer = unet_instance.model.get_layer("conv2d_8")
    conv2d_8_layer.trainable = False
    batch_norm_8_layer = unet_instance.model.get_layer("batch_normalization_8")
    batch_norm_8_layer.trainable = False
    conv2d_9_layer = unet_instance.model.get_layer("conv2d_9")
    conv2d_9_layer.trainable = False
    batch_norm_9_layer = unet_instance.model.get_layer("batch_normalization_9")
    batch_norm_9_layer.trainable = False
    block5 = unet_instance.model.get_layer("batch_normalization_9").output
    # ---------------------------------------------------------------------
    # Decoder Network
    # ---------------------------------------------------------------------
    # 6th Block
    up6 = concatenate([layer_manager.upsampling_layer(feature_size * 8, block5), block4])
    conv6 = layer_manager.conv2d_layer(feature_size * 8, up6, name='conv2d_10')
    conv6 = layer_manager.dropout_layer(conv6)
    conv6 = layer_manager.conv2d_layer(feature_size * 8, conv6, name='conv2d_11')
    block6 = layer_manager.dropout_layer(conv6)
    # ---------------------------------------------------------------------
    # 7th Block
    up7 = concatenate([layer_manager.upsampling_layer(feature_size * 4, block6), block3])
    conv7 = layer_manager.conv2d_layer(feature_size * 4, up7, name='conv2d_12')
    conv7 = layer_manager.dropout_layer(conv7)
    conv7 = layer_manager.conv2d_layer(feature_size * 4, conv7, name='conv2d_13')
    block7 = layer_manager.dropout_layer(conv7)
    # ---------------------------------------------------------------------
    # 8th Block
    up8 = concatenate([layer_manager.upsampling_layer(feature_size * 2, block7), block2])
    conv8 = layer_manager.conv2d_layer(feature_size * 2, up8, name='conv2d_14')
    conv8 = layer_manager.dropout_layer(conv8)
    conv8 = layer_manager.conv2d_layer(feature_size * 2, conv8, name='conv2d_15')
    block8 = layer_manager.dropout_layer(conv8)
    # ---------------------------------------------------------------------
    # 9th Block
    up9 = concatenate([layer_manager.upsampling_layer(feature_size, block8), block1])
    conv9 = layer_manager.conv2d_layer(feature_size, up9, name='conv2d_16')
    conv9 = layer_manager.dropout_layer(conv9)
    conv9 = layer_manager.conv2d_layer(feature_size, conv9, name='conv2d_17')
    block9 = layer_manager.dropout_layer(conv9)
    # ---------------------------------------------------------------------
    conv10 = layer_manager.fully_con_layer(block9)
    # ---------------------------------------------------------------------
    model = Model(inputs=input_layer, outputs=conv10)
    return model

def transfer_resnet_model(params, num_bands, num_cls=2):
    """Create a UNet model considering specific model parameter

    Parameter:
    params: dict consisting of important model parameter
    num_bands: Number of available spectral bands
    num_cls: Number of classes
    """
    layer_manager = Layers(params, num_bands, num_cls)
    feature_size = params["starting_feature_size"]
    inputs = layer_manager.input_layer()

    # set ResNet50 as backbone/encoder, excluding top layer
    resnet50 = ResNet50(include_top=False, weights="imagenet", input_tensor=inputs)

    # Change three channel input too four channel input
    resnet_config = resnet50.get_config()
    h, w, c = 256, 256, 4
    resnet_config['layers'][0]['config']['batch_input_shape'] = (None, h, w, c)
    resnet_updated = Model.from_config(resnet_config)
    print(resnet_updated.summary())
    new_model_conv1_block1_wts = resnet_updated.layers[2].get_weights()[0]

    resnet_updated_config = resnet_updated.get_config()
    resnet_updated_layer_names = [resnet_updated_config['layers'][x]['name'] for x in range(len(resnet_updated_config['layers']))]
    first_conv_name = resnet_updated_layer_names[0]
    print(first_conv_name)


    #for layer in resnet50.layers:
    #    if layer.name in resnet_updated_layer_names:

            #if layer.get_weights() != []:
            #    target_layer = resnet_updated.get_layer(layer.name)

                #if layer.name in first_conv_name:
                #    weights = layer.get_weights()[0]
                #    biases = layer.get_weights()[1]

                #    average_wts = np.mean(weights, axis=-2).reshape((weights[:, :, -1:, :]).shape)
                #    print(average_wts.shape)
                #    exit()

    # ---------------------------------------------------------------------
    # Transfered ResNet Encoder Network
    # ---------------------------------------------------------------------
    # build encoding part using backbone model
    b1_layer = resnet50.get_layer("input_1")
    b1_layer.trainable = False
    block1 = resnet50.get_layer("input_1").output

    b2_layer = resnet50.get_layer("conv1_relu")
    b2_layer.trainable = False
    block2 = resnet50.get_layer("conv1_relu").output

    b3_layer = resnet50.get_layer("conv2_block3_out")
    b3_layer.trainable = False
    block3 = resnet50.get_layer("conv2_block3_out").output

    b4_layer = resnet50.get_layer("conv3_block4_out")
    b4_layer.trainable = False
    block4 = resnet50.get_layer("conv3_block4_out").output

    b5_layer = resnet50.get_layer("conv4_block6_out")
    b5_layer.trainable = False
    block5 = resnet50.get_layer("conv4_block6_out").output
    # ---------------------------------------------------------------------
    # Decoder Network
    # ---------------------------------------------------------------------
    # 6th Block
    up6 = concatenate([layer_manager.upsampling_layer(feature_size * 8, block5), block4])
    conv6 = layer_manager.conv2d_layer(feature_size * 8, up6, name='conv2d_10')
    conv6 = layer_manager.dropout_layer(conv6)
    conv6 = layer_manager.conv2d_layer(feature_size * 8, conv6, name='conv2d_11')
    block6 = layer_manager.dropout_layer(conv6)
    # ---------------------------------------------------------------------
    # 7th Block
    up7 = concatenate([layer_manager.upsampling_layer(feature_size * 4, block6), block3])
    conv7 = layer_manager.conv2d_layer(feature_size * 4, up7, name='conv2d_12')
    conv7 = layer_manager.dropout_layer(conv7)
    conv7 = layer_manager.conv2d_layer(feature_size * 4, conv7, name='conv2d_13')
    block7 = layer_manager.dropout_layer(conv7)
    # ---------------------------------------------------------------------
    # 8th Block
    up8 = concatenate([layer_manager.upsampling_layer(feature_size * 2, block7), block2])
    conv8 = layer_manager.conv2d_layer(feature_size * 2, up8, name='conv2d_14')
    conv8 = layer_manager.dropout_layer(conv8)
    conv8 = layer_manager.conv2d_layer(feature_size * 2, conv8, name='conv2d_15')
    block8 = layer_manager.dropout_layer(conv8)
    # ---------------------------------------------------------------------
    # 9th Block
    up9 = concatenate([layer_manager.upsampling_layer(feature_size, block8), block1])
    conv9 = layer_manager.conv2d_layer(feature_size, up9, name='conv2d_16')
    conv9 = layer_manager.dropout_layer(conv9)
    conv9 = layer_manager.conv2d_layer(feature_size, conv9, name='conv2d_17')
    block9 = layer_manager.dropout_layer(conv9)
    # ---------------------------------------------------------------------
    conv10 = layer_manager.fully_con_layer(block9)
    # ---------------------------------------------------------------------
    model = Model(inputs=inputs, outputs=conv10)
    return model
