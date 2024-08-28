import pathlib
import random
import os
import matplotlib.pyplot as plt
import tensorflow as tf
import numpy as np
import cv2
import tifffile as tiff
from skimage import exposure

#from transformation import augmentate

class Dataset(object):
    """A class used to create and use a dataset of satellite imagery

    Attributes:
    bands: List of available spectral bands
    num_cls: Number of classes
    patch_width: Width of image patches
    patch_height: Height of image patches
    home_path: Home path to train data
    dataset: Preprocessed dataset
    """

    def __init__(self, bands, num_cls, patch_width, patch_height, path):
        self.bands = bands
        self.num_cls = num_cls
        self.patch_width = patch_width
        self.patch_height = patch_height
        self.home_path = path
        self.dataset = None

    def create_dataset_np(self, num_samples=1):
        """Create a numpy dataset

        Parameter:
        num_samples: number of wanted samples to be loaded
        """
        print('Collect and process data...')
        image_ds, mask_ds = self.list_objects(num_samples)
        print('Collecting done!')
        #print('Augmentation started')
        #for image, mask in list(zip(image_ds, mask_ds)):
        #    augmentate(image, mask)
        #    print('augmented')

        #print(image_ds[1, :, :, -1:0:-1].shape)
        #for i in range(30):
        #    plt.subplot(1, 2, 1)
        #    plt.imshow(image_ds[i, :, :, :3], label='gt image')
        #    plt.axis('off')
        #    plt.subplot(1, 2, 2)
        #    plt.imshow(mask_ds[i, :, :, 1], label='gt mask', cmap='gray')
        #    plt.axis('off')
        #    plt.show()
        #exit()
        self.dataset = (image_ds, mask_ds)

    def create_dataset_tf(self, num_samples=1):
        """Create a tf dataset to be used for model training

        Parameter:
        num_samples: number of wanted samples to be loaded
        """
        print('Collect data paths...')
        image_ds = tf.data.Dataset.list_files(self.home_path + '/*/*/*/image.npy', shuffle=True)
        if num_samples > 1 and num_samples < int(image_ds.__len__()):
            print('Taken samples: ' + str(num_samples))
            image_ds = image_ds.take(num_samples)
        print('Collecting done!\n')
        tuple_ds = image_ds.map(self.link_masks)
        print('Masks linked to images\n')

        print('Processing and filtering the data...')
        train_ds = tuple_ds.map(
            lambda img, mask: tf.numpy_function(self.process_npy_data, [img, mask], [tf.float32, tf.float32]))
        print('Processing done!\n')
        train_ds = train_ds.map(self.define_tf_shape)

        for img, mask in train_ds.take(1):
            print('***Image: ', img.shape)
            print('***Mask: ', mask.shape)
            #augmentate(img, mask)
        self.dataset = train_ds

    def list_objects(self, num_samples):
        """List all image paths for dataset specific conditions

        Parameter:
        num_samples: Number of samples to be loaded
        """
        image_ds, mask_ds = self.list_files()
        order = list(range(len(image_ds)))
        random.shuffle(order)
        order = np.array(order)
        if order.shape[0] > num_samples and num_samples != 1:
            order = order[:num_samples]
        image_ds = list(np.array(image_ds)[order])
        mask_ds = list(np.array(mask_ds)[order])
        return self.get_array_ds(image_ds, mask_ds)

    def list_files(self):
        """List all npy-prepared image paths"""
        image_ds = list(pathlib.Path(self.home_path).rglob('*image.npy'))
        mask_ds = list(pathlib.Path(self.home_path).rglob('*mask.npy'))
        return image_ds, mask_ds

    def load_data_from_file(self, files):
        """Load all data from given file_names

        Parameter:
        files: npy-file_list containing data of each patch
        """
        images = []
        for img_path in files: #pathlib.Path(self.home_path).rglob('*image.npy'):
            load_bands = np.load(img_path, allow_pickle=True)
            images.append(load_bands)
            del load_bands
        image_ds = np.array(images).astype(np.float32)
        return image_ds

    def link_masks(self, img_path):
        """Attach mask-path to corresponding image-path

        Parameter:
        img_path: Path to image patch
        """
        mask_path = tf.strings.regex_replace(img_path, 'image.npy', 'mask.npy')
        return img_path, mask_path

    def process_npy_data(self, img_path, mask_path):
        """Load npy-files for image and mask

        Parameter:
        img_path: Path to image_patch npy-file
        mask_path: Path to cloud mask npy-file
        """
        img = self.get_spectral_bands_from_file(img_path.decode())
        mask = self.get_cloud_mask_from_file(mask_path.decode())
        return img, mask

    def get_spectral_bands_from_file(self, file):
        """Select all requested spectral bands from image patch and put them together

        Parameter:
        file: npy-file containing spectral bands of image patch
        """
        bands = np.load(file, allow_pickle=True)
        #bands = bands[:, :, 2::-1]
        if len(bands.shape) == 2:
            bands = np.expand_dims(bands, 2)
        elif len(bands.shape) == 3:
            bands = bands[:, :, np.array(self.bands)]
        return bands.astype(np.float32)

    def get_cloud_mask_from_file(self, file):
        """Extract all cloud labeled pixel from mask filters and combine them in one mask

        Parameter:
        file: npy-file containing filter masks of image patch
        """
        load_masks = np.load(file, allow_pickle=True)
        # S3: already prepared mask
        if len(load_masks.shape) == 2:
            stacked_masks = np.stack((np.logical_not(load_masks).astype(np.float32), load_masks), axis=2).astype(np.float32)
        # L8: 3 = thin clouds | 4 = thick clouds
        elif load_masks.shape[2] == 1:
            stacked_masks = np.stack((np.logical_not(load_masks[:, :, 0]).astype(np.float32), load_masks[:, :, 0]), axis=2).astype(np.float32)
        else:
            stacked_masks = np.stack((np.logical_or(np.logical_or(load_masks[:, :, 1] == 1.0, load_masks[:, :, 2] == 1.0), load_masks[:, :, 0] == 1.0).astype(np.float32),
                np.logical_or(load_masks[:, :, 3] == 1.0, load_masks[:, :, 4] == 1.0).astype(np.float32)), axis=2).astype(np.float32)
        return stacked_masks

    def reduce_masks(self, ds):
        """Redruce L8 masks to cloudy and clear

        Parameter:
        ds: Ground truth dataset
        """
        if len(self.bands) == 1:
            return np.stack((np.logical_not(ds).astype(np.float32), ds), axis=3)
        elif ds.shape[3] == 1:
            return np.stack((np.logical_not(ds).astype(np.float32), ds), axis=3).astype(np.float32)
        else:
            # 0 = filler | 1 = shadow | 2 = clear | 3 = thin clouds | 4 = thick clouds
            return np.stack((np.logical_or(np.logical_or(ds[:, :, :, 1] == 1.0, ds[:, :, :, 2] == 1.0), ds[:, :, :, 0] == 1.0).astype(np.float32),
                             np.logical_or(ds[:, :, :, 3] == 1.0, ds[:, :, :, 4] == 1.0).astype(np.float32)), axis=3)

    def get_array_ds(self, image_ds, mask_ds):
        """Load and process images and masks out of their files

        Parameter:
        image_ds: Image path list
        mask_ds: Mask path list
        """
        image_files = image_ds
        mask_files = mask_ds
        if len(self.bands) > 1:
            image_ds = self.load_data_from_file(image_files)
            mask_ds = self.reduce_masks(self.load_data_from_file(mask_files))
        else:
            image_ds = self.load_data_from_file(image_files).astype(np.float32)
            mask_ds = self.reduce_masks(self.load_data_from_file(mask_files)).astype(np.float32)
        return image_ds, mask_ds

    def normalize_data(self, img, mask):
        """Normalize the image data

        Parameter:
        img: Array of the image patch
        mask: Array of the cloud mask
        """
        return (img / 255.), mask

    def train_val_split(self, val_split=0.05, sample_size=1):
        """Split the dataset into train_ds and test_ds for given splitting and sample_size

        Parameter:
        val_split: Split ratio for val_ds --> train_ds size automatically
        sample_size: Dataset size constraint
        """
        num_samples = int(self.dataset.__len__())
        if sample_size > 1 and sample_size < num_samples:
            num_samples = sample_size
        elif sample_size != 1:
            print('Sample size constraint is too high')
            exit()
        if num_samples > 10000:
            train_size = num_samples - 1000
        else:
            train_size = int(num_samples * (1 - val_split))
        val_size = num_samples - train_size

        train_ds = self.dataset.take(train_size)
        val_ds = self.dataset.skip(train_size).take(val_size)
        #train_ds = (self.dataset[0][:train_size], self.dataset[1][:train_size])
        #val_ds = (self.dataset[0][train_size:], self.dataset[1][train_size:])
        return train_ds, val_ds

    def get_shape(self):
        """Print the shape of the extracted data"""
        print('***Image Shape***: ',
              np.load(self.dataset[0][0], allow_pickle=True)[:, :, np.array(self.spec_bands)-1].shape)
        mask = np.load(self.dataset[1][0], allow_pickle=True)
        print('***Mask Shape***: ',
            np.stack((np.logical_or(np.logical_or(mask[:, :, :, 1] == 1.0, mask[:, :, :, 2] == 1.0), mask[:, :, :, 0] == 1.0).astype(np.float32),
                np.logical_or(mask[:, :, :, 3] == 1.0, mask[:, :, :, 4] == 1.0).astype(np.float32)), axis=3).shape)

    def define_tf_shape(self, img, mask):
        """Define the shape of tensors cause of separation from tensorflow graph build by using tf.numpy_function

        Parameter:
        img: image tensor
        mask: related mask tensor
        """
        img.set_shape([self.patch_width, self.patch_height, len(self.bands)])
        mask.set_shape([self.patch_width, self.patch_height, self.num_cls])
        return img, mask

    def get_kfold_set(self, k_cross, cross_idx):
        """Create the k-th fold from dataset

        Parameter:
        k_cross: Number of dataset folds
        cross_idx: k-th cross index"""
        num_samples = int(self.dataset.__len__())
        fold_size = num_samples // k_cross
        train_ds = self.dataset.take(cross_idx * fold_size)
        train_ds = train_ds.concatenate(self.dataset.skip((cross_idx + 1) * fold_size).take((k_cross - (cross_idx + 1)) * fold_size))
        val_ds = self.dataset.skip(cross_idx * fold_size).take(1 * fold_size)

        #val_set = indices[i * fold_size: (i + 1) * fold_size]
        #train_set = np.concatenate(indices[:i * fold_size], indices[(i + 1) * fold_size:])
        return (train_ds, val_ds)

if __name__ == '__main__':
    #BANDS = [3, 2, 1, 10]
    #dataset = Dataset(BANDS, 2, 256, 256, '/Sentinel-3/Creodias')
    #dataset.create_dataset()
    #print(str(int(dataset.dataset.__len__())) + " samples in the dataset")
    #path = '/Biome_256_Small_pp_md/test/Snow_Ice'
    path = '/TUBIN_256_pp_md/train'
    for campaign in os.listdir(path):
        campaign_path = os.path.join(path, campaign)
        products = [os.path.join(campaign_path, prod) for prod in os.listdir(campaign_path)]
        for prod in products:
            patches = [os.path.join(prod, patch) for patch in os.listdir(prod)]
            for patch in patches:
                img_path = os.path.join(patch, 'image.npy')
                mask_path = os.path.join(patch, 'mask.npy')
                img = np.load(img_path, allow_pickle=True)
                mask = np.load(mask_path, allow_pickle=True)
                #img_eq = img[:, :, :3] * 1.4
                #print(img[:, :, :3].max(), img_eq.max())
                # img_eq = exposure.equalize_hist(img[:, :, :3])
                # img_eq = exposure.equalize_adapthist(img[:, :, :3], clip_limit=0.03)
                #print(img_path)
                print(img.shape)
                plt.subplot(1, 2, 1)
                plt.imshow(img[:, :, :3])
                plt.subplot(1, 2, 2)
                plt.imshow(mask)
                plt.show()
