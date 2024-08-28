from tensorflow import keras
from keras.optimizers import Adagrad, SGD, Adadelta, Adam, Nadam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras import backend as K
import numpy as np

def get_standard_params():
    """Return the for the problem standard hyperparameter"""
    return dict(BANDS=[0, 1, 2, 3],
                starting_feature_size=16, #12,
                activation='leaky_relu', #'relu'
                num_cls=2,
                dropout_rate=0.2, #0.1
                patch_size=256)


def get_optimizer(opt_string, lr):
    """Return the requested optimizer function

    Parameter:
    opt_string: Optimizer in string format
    lr: Learning rate to alter the weights smoothly
    """
    if opt_string == 'adadelta':
        return Adadelta(learning_rate=lr, decay=1e-6)
    elif opt_string == 'adagrad':
        return Adagrad(learning_rate=lr, decay=1e-6)
    elif opt_string == 'adam':
        return Adam(learning_rate=lr) #, decay=1e-6)
    elif opt_string == 'nadam':
        return Nadam(learning_rate=lr, decay=1e-6)
    else:
        return SGD(learning_rate=lr)


def get_loss(loss_string):
    """Return the requested loss function

    Parameter:
    loss_string: Loss function in string format
    """
    if loss_string == 'categorical_crossentropy':
        return CategoricalCrossentropy()
    elif loss_string == 'binary_crossentropy':
        return BinaryCrossentropy()
    elif loss_string == 'mIoU':
        return mIoU


def f1_score(y_true, y_pred):
    """Calculates the F1 score == dice coefficient

    Parameter:
    y_true: tensor of ground truth masks
    y_pred: tensor of predicted masks
    """
    smooth = 1
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)


def dice_loss(y_true, y_pred):
    """Calculates the loss originating from F1 score

    Parameter:
    y_true: tensor of ground truth masks
    y_pred: tensor of predicted masks
    """
    return 1 - f1_score(y_true, y_pred)


def mIoU(y_true, y_pred):
    """Calculates the Jaccard index / Mean Intersection over Union

    Parameter:
    y_true: tensor of ground truth masks
    y_pred: tensor of predicted masks
    """
    smooth = 1e-12
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2])
    iou = (intersection + smooth) / (union - intersection + smooth)
    return K.mean(iou)


def mIoU_loss(y_true, y_pred):
    """Calculates the loss originating from mIoU score

    Parameter:
    y_true: tensor of ground truth masks
    y_pred: tensor of predicted masks
    """
    return 1 - mIoU(y_true, y_pred)
