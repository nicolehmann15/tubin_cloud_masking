import random
import tensorflow as tf
import tensorflow_addons as tfa

def brightness(img, mask):
    """Adjust brightness of image to alter intensity values

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    random.seed(15)
    delta = random.choice([0.1,-0.1])
    img = tf.image.adjust_brightness(img, delta)
    return img, mask


def gamma_correction(img, mask):
    """Adjust gamma value of image to alter differences between darker and lighter pixels

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    random.seed(15)
    gamma = random.choice([1.1, 0.9])
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
    angles = [30.0, 45.0, 75.0]
    aug_img = tfa.image.rotate(img, angles, fill_mode='nearest')
    aug_mask = tfa.image.rotate(mask, angles, fill_mode='nearest')
    return aug_img, aug_mask