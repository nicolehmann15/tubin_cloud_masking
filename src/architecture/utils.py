import numpy as np


def fuse_one_hot(oh_arr):
    """Fuse the one-hot encoded dimension to one single array

    Parameter:
    oh_arr: one-hot encoded arrays for number of samples
    """
    if isinstance(oh_arr, list):
        oh_arr = np.array(oh_arr)
    fuse_arr = (oh_arr[:, :, :, 1] > oh_arr[:, :, :, 0]).astype(np.float32)
    return fuse_arr


def cloud_amount(ds, patch_size):
    """ Calculate the portion of clouds in the dataset rounded to two digits after the comma.

    Parameter:
    ds: dataset
    patch_size: size of image patches
    """
    cloud_pixel = np.count_nonzero(ds)
    num_pixels = patch_size * ds.shape[0]
    cloud_rate = round(cloud_pixel * 100 / num_pixels, 2)
    print('The percentage of cloud pixels is ' + str(cloud_rate) + '%')