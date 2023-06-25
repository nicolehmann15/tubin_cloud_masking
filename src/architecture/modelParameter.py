from tensorflow import keras
from keras.optimizers import Adagrad, SGD, Adadelta, Adam
from keras.losses import CategoricalCrossentropy, BinaryCrossentropy

def get_optimizer(opt_string, lr):
    """Return the requested optimizer function

    Parameter:
    opt_string: Optimizer in string format
    lr: Learning rate to alter the weights smoothly
    """
    # TODO: which other parameter for the optimizer???
    if opt_string == 'adadelta':
        return Adadelta(learning_rate=lr)
    elif opt_string == 'adagrad':
        return Adagrad(learning_rate=lr)
    elif opt_string == 'adam':
        return Adam(learning_rate=lr)
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
    elif loss_string == 'jacc_coef':
        return jaccard_coefficient()

def jaccard_coefficient():
    print('jaccard')