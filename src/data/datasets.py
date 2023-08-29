import sys
import tensorflow as tf
import numpy as np
import random

from .cleaning import filter_cloudless
from .transformation import augmentate, compare, smoothing, sharpening

class Dataset(object):
    """A class used to create and use a dataset of satellite imagery

    Attributes:
    dataset_name: Name of the dataset in use
    bands: List of available spectral bands
    num_cls: Number of classes
    patch_width: Width of image patches
    patch_height: Height of image patches
    home_path: home path to train data
    images: Data loaded from directory
    masks: Masks loaded from directory
    """

    def __init__(self, dataset_name, bands, num_cls, patch_width, patch_height, path):
        self.dataset_name = dataset_name
        self.bands = bands
        self.num_cls = num_cls
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.home_path = path
        self.dataset = None
        self.masks = None

    def create_datasets(self, dataset_name):
        """Create the dataset of given data paths

        Parameter:
        dataset_name: filter to load the correct data
        """
        # Landsat8
        if dataset_name == '38-95-Cloud-Data':
            print('38-95')
            data_format = 'tif'
        elif dataset_name == 'biome':
            print('biome')
            self.num_satellite_bands = 12
            self.get_ccava_data()
        elif dataset_name == 'sparcs':
            print('sparcs')
            data_format = 'png'
        # Sentinel2
        elif dataset_name == 'cloud mask catalogue':
            print('catalogue')
            data_format = 'npy'
        elif dataset_name == 'cloudsen12':
            print('cloudsen12')
            print('data in pkl file')
        elif dataset_name == 's2_ccs':
            print('s2_ccs')
            data_format = 'tif'
        # TUBIN
        elif dataset_name == 'tubin':
            print('tubin')
            data_format = 'tif'

    def create_dataset(self):
        """Create a dataset to be used for model training"""
        print('Listing all files...')
        images_ds = self.list_images()
        print('Listing done!\n')

        tuple_ds = images_ds.map(self.link_masks)
        print('Masks linked to images\n')

        print('Processing the data...')
        train_ds = tuple_ds.map(lambda img, mask: tf.numpy_function(self.process_npy_data, [img, mask], [tf.float32, tf.float32]))
        train_ds = train_ds.map(self.define_tf_shape)
        print('Processing done!\n')

        if self.dataset_name != 'biome':
            print('Normalize patches...')
            train_ds = train_ds.map(self.normalize_data)
            print('Normalization done!\n')

        # print(int(train_ds.__len__()))
        # print('Cleaning the data...')
        # train_ds = train_ds.filter(lambda img, mask: tf.numpy_function(filter_cloudless, [img, mask], tf.bool))
        # print('Cleaning done!\n')
        # print(int(train_ds.__len__()))

        #for img, mask in train_ds.take(3):
        #    aug, mask = augmentate(img, mask)
        #    compare(img, aug)

        for img, mask in train_ds.take(1):
            print('***Image: ', img.shape)
            print('***Mask: ', mask.shape)
        self.dataset = train_ds

    def list_images(self):
        """List all image paths for dataset specific conditions"""
        if self.dataset_name == 'biome':
            images_ds = tf.data.Dataset.list_files(self.home_path + '/*/*/*/image.npy', shuffle=True)
        else:
            images_ds = tf.data.Dataset()
        return images_ds

    def link_masks(self, img_path):
        """Attach mask-path to corresponding image-path

        Parameter:
        img_path: Path to image patch
        """
        if self.dataset_name == 'biome':
            mask_path = tf.strings.regex_replace(img_path, 'image.npy', 'mask.npy')
        else:
            mask_path = ''
        return img_path, mask_path

    def process_npy_data(self, img_path, mask_path):
        """Load npy-files for image and mask

        Parameter:
        img_path: Path to image_patch npy-file
        mask_path: Path to cloud mask npy-file
        """
        if self.dataset_name == 'biome':
            img = self.get_spectral_bands_from_file(img_path.decode())
            mask = self.get_cloud_mask_from_file(mask_path.decode())
        else:
            img = np.zeros((self.patch_width, self.patch_height, len(self.bands)), dtype=np.float32)
            mask = np.zeros((self.patch_width, self.patch_height, self.num_cls), dtype=np.float32)
        return img, mask

    def get_spectral_bands_from_file(self, file):
        """Select all requested spectral bands from image patch and put them together

        Parameter:
        file: npy-file containing spectral bands of image patch
        """
        load_bands = np.load(file, allow_pickle=True)
        combination = np.zeros((self.patch_width, self.patch_height, len(self.bands)), dtype=np.float32)
        for order, band in enumerate(self.bands):
            combination[:, :, order] = load_bands[:, :, band-1]
        return combination

    def get_cloud_mask_from_file(self, file):
        """Extract all cloud labeled pixel from mask filters and combine them in one mask

        Parameter:
        file: npy-file containing filter masks of image patch
        """
        load_masks = np.load(file, allow_pickle=True)
        width, height, _ = load_masks.shape
        # 3 = thin clouds | 4 = thick clouds
        cloud_cond = np.where((load_masks[:, :, 3] == 1) | (load_masks[:, :, 4] == 1))
        mask = np.zeros((width, height, self.num_cls), dtype=np.float32)
        cloudy = np.zeros((width, height), dtype=np.float32)
        clear = np.ones((width, height), dtype=np.float32)
        clear[cloud_cond] = 0
        cloudy[cloud_cond] = 1
        mask[:, :, 0] = clear
        mask[:, :, 1] = cloudy
        return mask

    def normalize_data(self, img, mask):
        """Normalize the image data

        Parameter:
        img: Array of the image patch
        mask: Array of the cloud mask
        """
        return (img / 255.), mask

    def train_val_split(self, val_split=0.05):
        """Split the dataset into train_ds and test_ds for given splitting

        Parameter:
        bands: Spectral bands to be used
        test_size: Split ratio for test_ds --> train_ds size automatically
        """
        num_samples = int(self.dataset.__len__())
        if num_samples > 10000:
            train_size = num_samples - 500
        else:
            train_size = int(num_samples * (1 - val_split))
        val_size = num_samples - train_size

        train_ds = self.dataset.take(train_size)
        val_ds = self.dataset.skip(train_size).take(val_size)

        return train_ds, val_ds

    def define_tf_shape(self, img, mask):
        """Define the shape of tensors cause of separation from tensorflow graph build by using tf.numpy_function"""
        img.set_shape([self.patch_width, self.patch_height, len(self.bands)])
        mask.set_shape([self.patch_width, self.patch_height, self.num_cls])
        return img, mask

    def cloud_amount(self):
        """ Calculate the portion of clouds in the dataset rounded to two digits after the comma."""
        _, counts = np.unique(self.train_masks, return_counts=True)
        cloud_rate = counts[1] / sum(counts)
        cloud_rate = round(cloud_rate * 10000) / 100
        print('The percentage of cloud pixels is ' + str(cloud_rate) + '%')


if __name__ == '__main__':
    BANDS = [3, 2, 1, 11]
    dataset = Dataset('biome', BANDS, 2, 256, 256, 'D:/Clouds/data/LandSat8/Biome_Small')
    dataset.create_dataset()
    print(str(int(dataset.dataset.__len__())) + " samples in the dataset")

    import numpy as np
    import matplotlib.pyplot as plt
    import tifffile as tiff

    img = tiff.imread(
        'E:/MA-Clouds/Praxis/cloudy_data/LandSat8/38-Cloud-Data/38-Cloud_training/train_gt/gt_patch_69_4_by_6_LC08_L1TP_032029_20160420_20170223_01_T1.tif')
    img_array = np.array(img)
    print(img_array.shape)
    plt.imshow(img)