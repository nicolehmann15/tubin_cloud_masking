from tensorflow import keras
from keras.optimizers import Adagrad, SGD, Adadelta, Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy
from keras import backend as K
import numpy as np

lr = 0.001
decay_rate = lr * 100

def get_optimizer(opt_string, lr):
    """Return the requested optimizer function

    Parameter:
    opt_string: Optimizer in string format
    lr: Learning rate to alter the weights smoothly
    """
    if opt_string == 'adadelta':
        return Adadelta(learning_rate=lr) #, weight_decay=decay_rate, ema_momentum=momentum)
    elif opt_string == 'adagrad':
        return Adagrad(learning_rate=lr) #, weight_decay=decay_rate, ema_momentum=momentum)
    elif opt_string == 'adam':
        return Adam(learning_rate=lr) #, weight_decay=decay_rate, ema_momentum=momentum)
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
    """Calculates the F1 score

    Parameter:
    y_true: tensor of ground truth masks
    y_pred: tensor of predicted masks
    """
    smooth = 1
    intersection = K.sum(y_true * y_pred, axis=[0, -1, -2])
    union = K.sum(y_true + y_pred, axis=[0, -1, -2])
    dice = (2. * intersection + smooth) / (union + smooth)
    return K.mean(dice)


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

def exp_decay(epoch):
    new_lr = lr * np.exp(-decay_rate * epoch)
    return new_lr
