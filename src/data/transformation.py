import random
import subprocess

import matplotlib.gridspec
import matplotlib.pyplot as plt
import tensorflow as tf
import tensorflow_addons as tfa
import cv2
import numpy as np
import shutil
import os
from PIL import Image

ESRGAN_Path = 'C:/Users/n_leh/Desktop/Masterarbeit/Repos/ESRGAN'
SCALE_FACTOR = 256*256 - 1

def augmentate(img, mask):
    """Apply Augmentation randomly on img/mask

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    # choices = [0, 0, 0, 0, 0, 0]
    old_img = img
    # img, mask = translation(img, mask)
    # random.seed(15) --> causes everytime the same result
    # choice = tf.random.uniform([1, 1], minval=0, maxval=1, dtype=tf.dtypes.int32)
    if tf.random.uniform([1, 1], minval=0, maxval=1, dtype=tf.dtypes.int32)[0, 0] == 1: # random.choice([0, 1]) == 1:
        # choices[0] = 1

        # alter intensity values
        if tf.random.uniform([1, 1], minval=0, maxval=1, dtype=tf.dtypes.int32)[0, 0] == 1:# random.choice([0, 1]) == 1:
            # choices[1] = 1
            img, mask = brightness(img, mask)

        # flip the patch
        flippy = tf.random.uniform([1, 1], minval=0, maxval=2, dtype=tf.dtypes.int32)[0, 0] # # random.choice([0, 1, 2])
        if flippy == 1:
            # choices[1] = 1
            img, mask = flipH(img, mask)
        elif flippy == 2:
            # choices[1] = 2
            img, mask = flipV(img, mask)

        # rotate randomly
        if tf.random.uniform([1, 1], minval=0, maxval=1, dtype=tf.dtypes.int32)[0, 0] == 1:
            # choices[3] = 1
            img, mask = rotate(img, mask)

        noisy = tf.random.uniform([1, 1], minval=0, maxval=2, dtype=tf.dtypes.int32)[0, 0] # random.choice([0, 1, 2])
        if noisy == 1:
            # choices[4] = 1
            img, mask = white_noise(img, mask)
        elif noisy == 2:
            # choices[4] = 2
            img, mask = salt_and_pepper(img, mask)

        # blur or ensharpen the patch
        sharp = tf.random.uniform([1, 1], minval=0, maxval=2, dtype=tf.dtypes.int32)[0, 0] # random.choice([0, 1, 2])
        if sharp == 1:
            # choices[5] = 1
            img, mask = smoothing(img, mask)
        elif sharp == 2:
            # choices[5] = 2
            img, mask = sharpening(img, mask)
    # compare(old_img, img)

    # gs = matplotlib.gridspec.GridSpec(3, 3)
    # plt.subplot(gs[0, 0])
    # plt.imshow(old_img[:, :, :3])
    # plt.title('Original')
    # plt.axis('off')

    # plt.subplot(gs[0, 1])
    # plt.imshow(img1[:, :, :3])
    # plt.title('Brightness')
    # plt.axis('off')

    # plt.subplot(gs[0, 2])
    # plt.imshow(img2[:, :, :3])
    # plt.title('Horizontal Flip')
    # plt.axis('off')

    # plt.subplot(gs[1, 0])
    # plt.imshow(img3[:, :, :3])
    # plt.title('Vertical Flip')
    # plt.axis('off')

    # plt.subplot(gs[1, 1])
    # plt.imshow(img4[:, :, :3])
    # plt.title('Rotation')
    # plt.axis('off')

    # plt.subplot(gs[1, 2])
    # plt.imshow(img5[:, :, :3])
    # plt.title('White Noise')
    # plt.axis('off')

    # plt.subplot(gs[2, 0])
    # plt.imshow(img6[:, :, :3])
    # plt.title('Salt&Pepper Noise')
    # plt.axis('off')

    # plt.subplot(gs[2, 1])
    # plt.imshow(img7[0, :, :, :3])
    # plt.title('Smoothing')
    # plt.axis('off')

    # plt.subplot(gs[2, 2])
    # plt.imshow(img8[0, :, :, :3])
    # plt.title('Sharpening')
    # plt.axis('off')

    # plt.tight_layout()
    # plt.show()
    return img, mask

def compare(img, aug):
    """Plot the image together with his augmented version

    Parameter:
    img: Original image patch
    aug: Augmented image patch
    """
    plt.subplot(2, 1, 1)
    plt.imshow(img[:, :, 3:0:-1])
    plt.title('Original Image RGB')
    plt.subplot(2, 1, 2)
    plt.imshow(aug[0, :, :, 3:0:-1])
    plt.title('Augmentated Image RGB')
    plt.tight_layout()
    plt.show()

def translation_npy(img, mask):
    shift = random.choice(list(range(5, 51)))
    direction = random.choice(['right', 'down', 'left', 'up'])
    aug = img.copy()
    if direction == 'right':
        right_slice = aug[:, -shift:].copy()
        aug[:, shift:] = aug[:, :-shift]
        aug[:, :shift] = right_slice
    if direction == 'left':
        left_slice = aug[:, :shift].copy()
        aug[:, :-shift] = aug[:, shift:]
        aug[:, -shift:] = left_slice
    if direction == 'down':
        down_slice = aug[-shift:, :].copy()
        aug[shift:, :] = aug[:-shift, :]
        aug[:shift, :] = down_slice
    if direction == 'up':
        upper_slice = aug[:shift, :].copy()
        aug[:-shift, :] = aug[shift:, :]
        aug[-shift:, :] = upper_slice
    plt.subplot(1, 2, 1)
    plt.imshow(img)
    plt.title('original')
    plt.subplot(1, 2, 2)
    plt.imshow(aug)
    plt.title('augmented')
    plt.show()
    return img, mask


def brightness(img, mask):
    """Adjust brightness of image to alter intensity values

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    offset = tf.random.uniform([1, 1], minval=-10, maxval=10, dtype=tf.dtypes.int32)[0][0] # random.randrange(0, 10, 1)
    delta = 0.05 + tf.cast(offset, tf.float32) * 0.01,
    img = tf.image.adjust_brightness(img, delta)
    return img, mask


def gamma_correction(img, mask):
    """Adjust gamma value of image to alter differences between darker and lighter pixels

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    factor = tf.random.uniform([1, 1], minval=0, maxval=1, dtype=tf.dtypes.int32)[0][0]
    gamma = 0.9 + tf.cast(factor, tf.float32) * 0.2 # random.choice([1.1, 0.9])
    img = tf.image.adjust_gamma(img, gamma)
    return img, mask


def crop(img, mask):
    """Crop image and mask to specific size

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    # crop both image and mask identically
    img = tf.image.central_crop(img, 0.7)
    # resize after cropping
    img = tf.image.resize(img, (128,128))
    mask = tf.image.central_crop(mask, 0.7)
    # resize afer cropping
    mask = tf.image.resize(mask, (128,128))
    # cast to integers as they are class numbers
    mask = tf.cast(mask, tf.uint8)
    return img, mask


def flipH(img, mask):
    """Flip horizontally image and mask identically

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    img = tf.image.flip_left_right(img)
    mask = tf.image.flip_left_right(mask)
    return img, mask


def flipV(img, mask):
    """Flip vertically image and mask identically

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    img = tf.image.flip_up_down(img)
    mask = tf.image.flip_up_down(mask)
    return img, mask


def rotate(img, mask):
    """Rotate the image and mask for three angles

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    # angles = [1, 2, 3]
    rot = tf.random.uniform([1, 1], minval=1, maxval=3, dtype=tf.dtypes.int32)[0][0] # random.choice(angles)
    aug_img = tf.image.rot90(img, rot)
    aug_mask = tf.image.rot90(mask, rot)
    return aug_img, aug_mask


def white_noise(img, mask):
    """Adds white noise to image

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    sigma = 0.05 # Standard deviation of white noise
    noise = tf.random.normal(shape=tf.shape(img), mean=0.0, stddev=sigma, dtype=tf.float32)
    return img + noise, mask


def salt_and_pepper(img, mask):
    """Adds salt and pepper (light and dark) noise to image

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    salt_rate = 0.02 # Percentage of pixels that are set to salt_value
    pepp_rate = 0.02 # Percentage of pixels that are set to pepp_value
    pepp_value = 0.2 # Value that pepper pixels are set to
    salt_value = 0.8 # Value that salt pixels are set to
    random_values = tf.random.uniform(shape=tf.shape(img)) # shape=img[..., -1:].shape
    aug_img = tf.where(random_values < salt_rate, salt_value, img)
    aug_img = tf.where(1 - random_values < pepp_rate, pepp_value, aug_img)
    return aug_img, mask


def smoothing(img, mask):
    """Blurs the image patch with Gaussian Filter Kernel

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    kernel_size = 5
    sigma = 3
    image = tfa.image.gaussian_filter2d(
        img,
        filter_shape=(kernel_size, kernel_size),
        sigma=sigma,
        padding='CONSTANT'
    )
    return image, mask

def smoothing2(img, mask):
    """Blurs the image patch with Gaussian Filter Kernel

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    kernel_size = 5
    sigma = 3
    #img = tf.expand_dims(img, 0) # testing inside dataset-creation

    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], tf.constant([1, 1, channels])) # channels.numpy() for testing inside dataset-creation
        return kernel

    gaussian_kernel = gauss_kernel(img.shape[-1], kernel_size=kernel_size, sigma=sigma)
    tf.print(gaussian_kernel)
    #gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1], padding='SAME', data_format='NHWC'), mask # result [0] for testing inside dataset-creation


def sharpening(img, mask):
    """Ensharpens the image patch with Laplacian-5 Filter Kernel

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """

    channels = img.shape[-1]
    img = tf.expand_dims(img, 0) # testing inside dataset-creation
    laplace_kernel = tf.constant([  [0, -1, 0], # Laplacian-5
                                    [-1, 5, -1],
                                    [0, -1, 0]], dtype=tf.float32)

    laplace_kernel = tf.tile(laplace_kernel[..., tf.newaxis], tf.constant([1, 1, channels])) # channels.numpy() for testing inside dataset-creation
    laplace_kernel = laplace_kernel[..., tf.newaxis]
    return tf.nn.depthwise_conv2d(img, laplace_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC'), mask # result [0] for testing inside dataset-creation


def rescaling(img, resolution, img_path=None):
    """Rescale an image with higher / lower change options

    Parameter:
    img: Image to be rescaled
    resolution: Requested image resolution
    img_path: path of image to be loaded
    """
    print(img.shape, resolution)
    ratio = img.shape[0] / resolution[0]
    print('ratio: ' + str(ratio))
    if ratio < 0.8:
        print('high change')
        dst_dir = os.path.join(ESRGAN_Path, 'LR')
        shutil.copyfile(img_path, os.path.join(dst_dir, os.path.basename(img_path)))
        subprocess.call(['python', os.path.join(ESRGAN_Path, 'test.py')])
        print('after call')
        result_path = os.path.join(ESRGAN_Path, 'result')
        rescaled = np.array(Image.open(result_path))
        return rescaled
    else:
        print('low change')
        new_img = []
        for channel in range(img.shape[2]):
            new_img.append(cv2.resize(img[:, :, channel], dsize=(resolution[1], resolution[0]), interpolation=cv2.INTER_CUBIC))
        return np.stack(new_img, axis=2)


def standardize_data(data_arr):
    """Standardize to interval [0.0, 1.0]

    Parameter:
    data_arr: Dataset array
    """
    data_arr = np.round(data_arr / SCALE_FACTOR, 5)
    return data_arr