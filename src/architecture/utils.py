import numpy as np
import matplotlib.pyplot as plt


def fuse_one_hot(oh_arr):
    """Fuse the one-hot encoded dimension to one single array

    Parameter:
    oh_arr: one-hot encoded arrays for number of samples
    """
    samples, width, height, _ = oh_arr.shape
    fuse_arr = np.zeros((samples, width, height), dtype=np.float32)
    cloudy = np.where(oh_arr[:, :, :, 1] == 1.0)
    fuse_arr[cloudy] = 1.0
    return fuse_arr
