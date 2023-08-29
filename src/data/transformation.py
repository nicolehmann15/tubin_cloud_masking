import random
import matplotlib.pyplot as plt
import tensorflow as tf

def augmentate(img, mask):
    """Apply Augmentation randomly on img/mask

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    random.seed(15)
    if random.choice([0, 1]) == 1:
        # alter intensity values
        if random.choice([0, 1]) == 1:
            img, mask = brightness(img, mask)

        # flip the patch
        flippy = random.choice([0, 1, 2])
        if flippy == 1:
            img, mask = flipH(img, mask)
        elif flippy == 2:
            img, mask = flipV(img, mask)

        # rotate randomly
        if random.choice([0, 1]) == 1:
            img, mask = rotate(img, mask)

        noisy = random.choice([0, 1, 2])
        if noisy == 1:
            img, mask = white_noise(img, mask)
        elif noisy == 2:
            img, mask = salt_and_pepper(img, mask)

        # blur or ensharpen the patch
        sharp = random.choice([0, 1, 2])
        if sharp == 1:
            img, noise = smoothing(img, mask)
        elif sharp == 2:
            img, mask = sharpening(img, mask)
    return (img, mask)

def compare(img, aug):
    """Plot the image together with his augmented version

    Parameter:
    img: Original image patch
    aug: Augmented image patch
    """
    plt.subplot(2, 1, 1)
    plt.imshow(img[:, :, :3])
    plt.title('Original Image RGB')
    plt.subplot(2, 1, 2)
    plt.imshow(aug[:, :, :3])
    plt.title('Augmentated Image RGB')
    plt.tight_layout()
    plt.show()

def brightness(img, mask):
    """Adjust brightness of image to alter intensity values

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
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
    angles = [1, 2, 3]
    rot = random.choice(angles)
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
    salt_rate =  0.025 # Percentage of pixels that are set to salt_value
    pepp_rate =  0.025 # Percentage of pixels that are set to pepp_value
    pepp_value = 0.0 # Value that pepper pixels are set to
    salt_value = 1.0 # Value that salt pixels are set to
    random_values = tf.random.uniform(shape=img[0, ..., -1:].shape)
    aug_img = tf.where(random_values < salt_rate, salt_value, img)
    aug_img = tf.where(1 - random_values < pepp_rate, pepp_value, aug_img)
    return aug_img, mask


def smoothing(img, mask):
    """Blurs the image patch with Gaussian Filter Kernel

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    kernel_size = 7
    sigma = 3
    # img = tf.expand_dims(img, 0) # testing inside dataset-creation

    def gauss_kernel(channels, kernel_size, sigma):
        ax = tf.range(-kernel_size // 2 + 1.0, kernel_size // 2 + 1.0)
        xx, yy = tf.meshgrid(ax, ax)
        kernel = tf.exp(-(xx ** 2 + yy ** 2) / (2.0 * sigma ** 2))
        kernel = kernel / tf.reduce_sum(kernel)
        kernel = tf.tile(kernel[..., tf.newaxis], tf.constant([1, 1, channels])) # channels.numpy() for testing inside dataset-creation
        return kernel

    gaussian_kernel = gauss_kernel(img.shape[-1], kernel_size=kernel_size, sigma=sigma)
    gaussian_kernel = gaussian_kernel[..., tf.newaxis]
    return tf.nn.depthwise_conv2d(img, gaussian_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC'), mask # result [0] for testing inside dataset-creation


def sharpening(img, mask):
    """Ensharpens the image patch with Laplacian-5 Filter Kernel

    Parameter:
    img: Image Patch to be augmented
    mask: Cloud mask of the image
    """
    channels = img.shape[-1]
    # img = tf.expand_dims(img, 0) # testing inside dataset-creation
    laplace_kernel = tf.constant([  [0, -1, 0], # Laplacian-5
                                    [-1, 5, -1],
                                    [0, -1, 0]], dtype=tf.float32)
    laplace_kernel = tf.tile(laplace_kernel[..., tf.newaxis], tf.constant([1, 1, channels])) # channels.numpy() for testing inside dataset-creation
    laplace_kernel = laplace_kernel[..., tf.newaxis]
    return tf.nn.depthwise_conv2d(img, laplace_kernel, [1, 1, 1, 1], padding='SAME', data_format='NHWC'), mask # result [0] for testing inside dataset-creation