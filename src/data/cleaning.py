import os.path
import pathlib
import numpy as np
import shutil

THRESH = 0.01
BANDS = [3, 2, 1, 10]

def filter_undesired(image_ds, mask_ds):
    """Remove all data from dataset including invalid data or fully clear data

    Parameter:
    image_ds: image dataset
    mask_ds: related mask dataset
    """
    take_list = []
    for i in range(image_ds.shape[0]):
        # invalid data  OR  not cloudy data
        if np.all(mask_ds[i, :, :, 0] == 1) or (np.all(mask_ds[i, :, :, 3] == 0) and np.all(mask_ds[i, :, :, 4] == 0)):
            continue
        take_list.append(i)
    print('invalid: ', str(image_ds.shape[0]), str(len(take_list)))
    return np.take(image_ds, take_list, axis=0), np.take(mask_ds, take_list, axis=0)


def clean_biome(home_path):
    """Delete all clear products

    Parameter:
    home_path: path to the dataset
    """
    i = 0
    deletions = []
    for mask_path in pathlib.Path(home_path).rglob('*mask.npy'):
        load_mask = np.load(mask_path, allow_pickle=True)
        if np.all(load_mask[:, :, 0] == 1):
            # print('remove dir: ', os.path.dirname(mask_path))
            deletions.append(os.path.dirname(mask_path))
        i += 1
        if i % 1000 == 0:
            print(str(i) + '. pair')
        del load_mask
    print(len(deletions))
    for i, path in enumerate(deletions):
        if i % 1000 == 0:
            print('deletion number: ', i)
        shutil.rmtree(path)


def cloud_amount(home_path, patch_size):
    """Compute the cloud amount of the ground truth dataset
        investigating the masks

    Parameter:
    home_path: path to the dataset
    patch_size: patch_size of the ground truth images
    """
    cnt = 0
    cloud_pixel = 0
    almost_clear = 0
    almost_cloudy = 0
    for mask_path in pathlib.Path(home_path).rglob('*mask.npy'):
        load_mask = np.load(mask_path, allow_pickle=True)
        # if not np.all(load_mask[:, :, 0] == 0):# and (np.any(load_mask[:, :, 3] == 1) or np.any(load_mask[:, :, 4] == 1)):
        cloudy = np.count_nonzero(load_mask) # np.count_nonzero(load_mask[:, :, 3]) + np.count_nonzero(load_mask[:, :, 4])
        cloud_rate = (cloudy) / (patch_size * patch_size)
        if cloud_rate < 0.05:
            #print('almost clear')
            almost_clear += 1
        if cloud_rate > 0.95:
            almost_cloudy += 1
        cloud_pixel += cloudy
        cnt += 1
        del load_mask
    print('cloudy patches: ', cnt)
    print('almost clear: ', almost_clear)
    print('almost cloudy: ', almost_cloudy)
    pixel_cnt = cnt * patch_size * patch_size
    print('Cloud Pixels: ', cloud_pixel)
    print('Pixel Amount: ', pixel_cnt)
    print(f'The percentage of cloud pixels is {round(100 * cloud_pixel / pixel_cnt, 2)}%')


if __name__ == '__main__':
    cloud_amount('/TUBIN_256_pp_md/train', 256)
    #clean_biome('/Landsat8/Processing/patched')
