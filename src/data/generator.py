import numpy as np
import pathlib
import keras
import tensorflow as tf

class CustomGenerator(keras.utils.Sequence):
    """A class used to create and use a data generator for satellite imagery

    Attributes:
    image_ds: List of image paths
    mask_ds: List of mask paths
    batch_size: Number of samples per minibatch
    spec_bands: List of bands prioterized
    """

    def __init__(self, image_ds, mask_ds, batch_size, spec_bands):
        self.image_ds = image_ds
        self.mask_ds = mask_ds
        self.batch_size = batch_size
        self.spec_bands = spec_bands

    def __len__(self):
        """Compute the size of the dataset"""
        return (np.ceil(len(self.image_ds) / float(self.batch_size))).astype('int32')

    def __getitem__(self, idx):
        """Load an process a minibatch to be used in training

        Parameter:
        idx: Index of wanted batch of dataset
        """
        batch_x = self.image_ds[idx * self.batch_size: (idx + 1) * self.batch_size]
        batch_y = self.mask_ds[idx * self.batch_size: (idx + 1) * self.batch_size]

        batch_arr_x = self.load_data_from_files(batch_x)[:, :, np.array(self.spec_bands)-1]
        masks = self.load_data_from_files(batch_y)
        # 1 = shadow | 2 = clear | 3 = thin clouds | 4 = thick clouds
        batch_arr_y = np.stack((np.logical_or(masks[:, :, :, 1] == 1.0, masks[:, :, :, 2] == 1.0).astype('float32'),
                         np.logical_or(masks[:, :, :, 3] == 1.0, masks[:, :, :, 4] == 1.0).astype('float32')), axis=3)
        return batch_arr_x, batch_arr_y

    def load_data_from_files(self, files):
        """Select all requested spectral bands from image patch and put them together

        Parameter:
        files: npy-file path containing spectral bands of image patch
        """
        images = []
        for img_path in files:
            load_bands = np.load(img_path, allow_pickle=True)
            images.append(load_bands)
            del load_bands
        image_ds = np.array(images)
        return image_ds