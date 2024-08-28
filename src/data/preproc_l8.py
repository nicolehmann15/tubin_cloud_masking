import os
import pathlib
import shutil
import numpy as np
import cv2
import matplotlib.pyplot as plt
import tifffile as tiff
import spectral as spy

from datasets import Dataset

LANDSAT_PRED_PATH = 'D:/Clouds/data/test_predict'
LANDSAT_PATH = 'D:/Clouds/data/Landsat8/Processing/buffer'
LANDSAT_PP = 'D:/Clouds/data/Landsat8/Biome_256_pp'
LANDSAT_PP_MD = 'D:/Clouds/data/Landsat8/Biome_256_pp_md'
LANDSAT_FULL = 'D:/Clouds/data/Landsat8/Biome_Full'
SCALE_FACTOR = 256*256 - 1
BAND_ORDER = [3, 2, 1, 0]
THRESH = 0.01
KEEP_BANDS = ["B2", "B3", "B4", "B10", 'fixedmask']
DELIMITER = '_'

def drop_files(home_path):
    biome_list = os.listdir(home_path)
    for biome in biome_list:
        biome_path = os.path.join(home_path, biome)
        product_list = os.listdir(biome_path)
        for product in product_list:
            product_path = os.path.join(biome_path, product)
            files = os.listdir(product_path)
            for file in files:
                band = file.split(DELIMITER)[-1].split('.')[0]
                if band not in KEEP_BANDS:
                    file_path = os.path.join(product_path, file)
                    os.remove(file_path)


def filter_undesired(image_ds, mask_ds):
    """Remove all data from dataset including invalid data or fully clear data

    Parameter:
    image_ds: image dataset
    mask_ds: related mask dataset
    """
    take_list = []
    for i in range(image_ds.shape[0]):
        # invalid data  OR  not cloudy data
        if np.any(mask_ds[i, :, :, 0] == 1) or (np.all(mask_ds[i, :, :, 3] == 0) and np.all(mask_ds[i, :, :, 4] == 0)):
            continue
        take_list.append(i)
    print('invalid: ', str(image_ds.shape[0]), str(len(take_list)))
    return np.take(image_ds, take_list, axis=0), np.take(mask_ds, take_list, axis=0)


def clean_biome(home_path):
    """Delete all clear products

    Parameter:
    home_path: path to the dataset
    """
    deletions = []
    for mask_path in pathlib.Path(home_path).rglob('*mask.npy'):
        load_mask = np.load(mask_path, allow_pickle=True)
        if np.any(load_mask[:, :, 0] == 1):
            # print('remove dir: ', os.path.dirname(mask_path))
            deletions.append(os.path.dirname(mask_path))
        if len(deletions) % 1000 == 0:
            print('Patch number: ' + str(len(deletions)))
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
        if np.all(load_mask[:, :, 0] == 0):# and (np.any(load_mask[:, :, 3] == 1) or np.any(load_mask[:, :, 4] == 1)):
            cloudy = np.count_nonzero(load_mask[:, :, 3]) + np.count_nonzero(load_mask[:, :, 4])
            cloud_rate = (cloudy) / (patch_size * patch_size)
            cnt += 1
            if cnt % 1000 == 0:
                print('Patch number: ' + str(cnt))
            if cloud_rate < 0.05:
                #print('almost clear')
                almost_clear += 1
            if cloud_rate > 0.95:
                #print('nearly full of clouds')
                almost_cloudy += 1
            cloud_pixel += cloudy
        del load_mask
    print('cloudy patches: ', cnt)
    print('almost clear: ', almost_clear)
    print('almost cloudy: ', almost_cloudy)
    pixel_cnt = cnt * patch_size * patch_size
    print('Cloud Pixels: ', cloud_pixel)
    print('Pixel Amount: ', pixel_cnt)
    print(f'The percentage of cloud pixels is {round(100 * cloud_pixel / pixel_cnt, 2)}%')


def process_dataset(home_path):
    """Extract all file paths for given home_path and save them after processing and splitting

    Parameter:
    home_path: Starting point to search for files
    """
    biome_list = os.listdir(home_path)
    for biome in biome_list:
        print('\nBiome: ' + biome)
        #if biome != 'Urban':
            #continue
        biome_path = os.path.join(home_path, biome)
        biome_dir = biome_path.replace('/buffer', '/patched')
        if os.path.isdir(biome_dir) is False:
            os.mkdir(biome_dir)
        processed_path = biome_path.replace('/buffer', '/processed')
        if os.path.isdir(processed_path) is False:
            os.mkdir(processed_path)
        product_list = os.listdir(biome_path)
        for product in product_list:
            product_path = os.path.join(biome_path, product)
            band_files = [os.path.join(product_path, file) for file in os.listdir(product_path) if not file.endswith('.img') and not file.endswith('.hdr')]
            mask_path = [os.path.join(product_path, file) for file in os.listdir(product_path) if file.endswith('.hdr')][0]
            product_arr = collect_data_from_files(band_files)
            print(product_arr.shape)
            #plt.subplot(1, 2, 1)
            plt.imshow(standardize_data(product_arr[:, :, 3:0:-1]))
            product_arr = product_arr[:, :, np.array(BAND_ORDER)]
            #plt.subplot(1, 2, 2)
            #plt.imshow(standardize_data(product_arr[:, :, :3])) --> ':3' ist die richtige ORDER
            #plt.show()
            #continue
            #print(np.unique(product_arr, return_counts=True))
            # fill value: 0, smallest values are starting at 7772 --> wrong standardization (new dataset generation)
            mask_arr = np.squeeze(spy.open_image(mask_path).load())
            # plt.imshow(product_arr[:, :, 0])
            # plt.show()
            product_mask = np.zeros((mask_arr.shape[0], mask_arr.shape[1], 5), dtype=np.float32)
            product_mask[..., 0] = (mask_arr == 0)    # FILL
            product_mask[..., 1] = (mask_arr == 64)   # SHADOW
            product_mask[..., 2] = (mask_arr == 128)  # CLEAR
            product_mask[..., 3] = (mask_arr == 192)  # THIN
            product_mask[..., 4] = (mask_arr == 255)  # THICK
            resized_product = resize_to_patchify(product_arr)
            patched_product = sliding_window(resized_product[0], win_size=256, stride=256)
            patched_product = standardize_data(patched_product)
            resized_mask = resize_to_patchify(product_mask)
            patched_mask = sliding_window(resized_mask[0], win_size=256, stride=256)
            #print(patched_product.shape)
            # for idx in range(patched_mask.shape[0]):
            #    plt.subplot(2, 5, 1)
            #    plt.imshow(patched_img[idx, : , :, 0], cmap='gray', vmin=0, vmax=1)
            #    plt.subplot(2, 5, 6)
            #    plt.imshow(patched_mask[idx, :, :, 0], cmap='gray', vmin=0, vmax=1)
            #    plt.subplot(2, 5, 7)
            #    plt.imshow(patched_mask[idx, :, :, 1], cmap='gray', vmin=0, vmax=1)
            #    plt.subplot(2, 5, 8)
            #    plt.imshow(patched_mask[idx, :, :, 2], cmap='gray', vmin=0, vmax=1)
            #    plt.subplot(2, 5, 9)
            #    plt.imshow(patched_mask[idx, :, :, 3], cmap='gray', vmin=0, vmax=1)
            #    plt.subplot(2, 5, 10)
            #    plt.imshow(patched_mask[idx, :, :, 4], cmap='gray', vmin=0, vmax=1)
            # plt.show()
            product_dir = product_path.replace('/buffer', '/patched')
            if os.path.isdir(product_dir) is False:
                os.mkdir(product_dir)
            for patch_idx in range(1, patched_product.shape[0] + 1):
                patch = '00' + str(patch_idx)
                if patch_idx < 10:
                    patch = '0' + patch
                elif patch_idx >= 100:
                    if patch_idx >= 1000:
                        patch = patch[2:]
                    else:
                        patch = patch[1:]
                patch_dir = os.path.join(product_dir, patch)
                if os.path.isdir(patch_dir) is False:
                    os.mkdir(patch_dir)
                np.save(os.path.join(patch_dir, 'image.npy'), patched_product[patch_idx - 1])
                np.save(os.path.join(patch_dir, 'mask.npy'), patched_mask[patch_idx - 1])
            processed_product = product_path.replace('/buffer', '/processed')
            shutil.move(product_path, processed_product)


def process_dir(home_path):
    """Extract all prediction-file paths for given home_path

    Parameter:
    home_path: Starting point to search for files
    """
    product_list = os.listdir(home_path)
    data_arr = []
    origin_dict = {'path': [], 'og_size': 0, 'resized_res': 0}
    for product in product_list:
        product_path = os.path.join(home_path, product)
        band_files = [os.path.join(product_path, file) for file in os.listdir(product_path)]
        product_arr = collect_data_from_files(band_files)
        product_arr = product_arr[:, : , np.array(BAND_ORDER)]
        data_arr.append(product_arr)
        origin_dict['path'].append(product_path)
    data_arr = np.stack(data_arr, axis=0)
    origin_dict['og_size'] = data_arr[0].shape[0:2]
    data_arr = standardize_data(data_arr)
    return data_arr, origin_dict


def collect_data_from_files(list_files):
    """Load data from tif(f) files

    Parameter:
    list_files: List of tif(f) files
    """
    data_products = []
    print(list_files)
    for path in [file for file in list_files if file.endswith('.TIF') and 'B8' not in file]:
        #print(path)
        img = tiff.imread(path)
        img_array = np.array(img).astype(np.float32)
        data_products.append(img_array)
    data_products = np.stack(data_products, axis=-1)
    #print(np.min(data_products[:, :, 0]), np.max(data_products[:, :, 0]))
    #exit()
    return data_products
    #plt.imshow(standardize_data(data_products[:, :, 5:2:-1]))
    tir_img = standardize_data(data_products[:, :, 1])
    mask_path = list_files[0].rsplit('_B1.TIF', 1)
    mask_path = '_fixedmask.hdr'.join(mask_path)
    mask_arr = np.squeeze(spy.open_image(mask_path).load())
    mask_arr = np.logical_or(mask_arr == 192, mask_arr == 255)
    #for i in range(0, 11, 1):
    #    thresh = np.logical_and(tir_img > 0.0, tir_img < (0.308 + i*0.001))
    #    compare = thresh != mask_arr
    #    print(str(round(np.sum(compare) * 100 / (tir_img.shape[0] * tir_img.shape[1]), 3)) + '% Fehler bei Threshold ' + str(0.308 + i*0.001))
    #    plt.imshow(compare, cmap='gray')
    #    plt.show()
    # THRESHOLD 0.312 is best to get best pre-mask
    #plt.subplot(2, 2, 1)
    #plt.imshow(tir_img, cmap='gray')
    #thresh = np.logical_and(tir_img > 0.0, tir_img < 0.29)
    #plt.subplot(2, 2, 1)
    #plt.subplot(2, 2, 2)
    #plt.imshow(thresh, cmap='gray')
    #plt.subplot(2, 2, 3)
    #plt.imshow(mask_arr, cmap='gray')
    #plt.show()
    return data_products


def standardize_data(data_arr):
    """Standardize to interval [0.0, 1.0]

    Parameter:
    data_arr: Dataset array
    """
    data_arr = np.round(data_arr / SCALE_FACTOR, 5)
    return data_arr


def resize_to_patchify(data_arr):
    """Resize product to sliceable size (256*X)

    Parameter:
    data_arr: Dataset array
    """
    if len(data_arr.shape) == 3:
        data_arr = np.expand_dims(data_arr, axis=0)
    arr_shape = data_arr.shape
    dim_size_x = round(arr_shape[1] / 256) * 256
    dim_size_y = round(arr_shape[2] / 256) * 256
    resolution = (dim_size_y, dim_size_x)
    resized_arr = []
    for product in range(arr_shape[0]):
        resized_arr.append(cv2.resize(data_arr[product], resolution, interpolation=cv2.INTER_LINEAR_EXACT))#INTER_AREA))
    return np.stack(resized_arr, axis=0)


def show_all_patches(data_arr):
    """Show all patches in array

    Parameter:
    data_arr: Array of Dataset
    """
    num_patches = data_arr.shape[2]
    for index in range(1, num_patches+1):
        plt.subplot(1, num_patches, index)
        plt.imshow(data_arr[:, :, index-1], cmap='gray')
        plt.title('Data product\'s ' + str(index) + 'th item')
    plt.show()


def slice_data(image_arr):
    """Slice the dataset dependent on the task (preprocessing/prediction)

    Parameter:
    image_arr: Array of image products
    """
    for idx in range(image_arr.shape[0]):
        sliced_data = sliding_window(image_arr[idx], win_size=256, stride=256)
        return sliced_data


def sliding_window(data, win_size, stride=None):
    """Use a sliding window to iterate over an array with given window size

    Parameter:
    data: Data array
    win_size: Size of sliding window
    stride: Number of skiped pixels after each taken window snapshot
    """
    if stride == None:
        stride = win_size
    id_x = 0
    width, height, _ = data.shape
    slices = []
    while id_x + win_size <= width:
        id_y = 0
        while id_y + win_size <= height:
            slices.append(data[id_x:id_x+win_size, id_y:id_y+win_size])
            id_y += stride
        id_x += stride
    patched_data = np.stack(slices, axis=0)
    return patched_data


def join_patches(origin_dict, pred_arr):
    """Recreate original image structure with
        predicted cloud masks of image patches

    Parameter:
    origin_dict: Dictionary with information about original image
    pred_arr: Array of cloud masks
    """
    num_patches = pred_arr.shape[0]
    og_width, og_height = origin_dict['og_size']
    resized_height, resized_width = origin_dict['resized_res']
    patches_per_img = round((resized_width * resized_height) / (256 * 256))
    num_masks = round(num_patches / patches_per_img)
    mask_arr = np.zeros((num_masks, og_width, og_height))
    for product in range(num_masks):
        joined_product = joining_window(pred_arr[product*patches_per_img: (product+1)*patches_per_img], origin_dict['resized_res'])
        mask_arr[product] = cv2.resize(joined_product, (og_height, og_width), interpolation=cv2.INTER_AREA)
    print(mask_arr.shape)
    return mask_arr


def joining_window(data, target_res):
    """Join all patches using a sliding window

    Parameter:
    data: Array of mask patches
    target_res: Resolution of original image
    """
    print(data.shape)
    _, win_size, stride = data.shape
    width, height = target_res
    joined_arr = np.zeros((width, height))
    id_x = 0
    id_y = 0
    for idx in range(data.shape[0]):
        if id_x + win_size > width:
            id_x = 0
            id_y += stride
        joined_arr[id_x:id_x+win_size, id_y:id_y+win_size] = data[idx]
        id_x += stride
    return joined_arr


def preprocess_data(home_path, task):
    """Preprocess a Landsat8 dataset

    Parameter:
    home_path: Starting point for file extraction
    task: preprocessing dataset / collect and prepare prediction data
    """
    if task == 'preprocessing':
        process_dataset(home_path)
    elif task == 'predict':
        data_arr, origin_dict = process_dir(home_path)
        resized = resize_to_patchify(data_arr)
        origin_dict['resized_res'] = resized[0].shape[0:2]
        print(resized.shape)
        sliced = slice_data(resized)
        print(sliced.shape)
        return sliced, origin_dict


if __name__ == '__main__':
    #preprocess_data(LANDSAT_PRED_PATH, 'predict')

    #drop_files(LANDSAT_PATH)
    #preprocess_data(LANDSAT_PATH, 'preprocessing')
    #preprocess_data(LANDSAT_PATH, 'preprocessing')

    #clean_biome(LANDSAT_PP)
    cloud_amount(LANDSAT_PP_MD, 256)