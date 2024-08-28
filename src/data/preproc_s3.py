import os
import pathlib
import numpy as np
import cv2
import matplotlib.pyplot as plt
import rasterio
import tifffile as tiff

CREODIAS_PATH = '/Sentinel-3/Creodias/original/'
SLSTR_SHAPE = (1500, 1200)
SLSTR_FILL_VALUE = -32768
S8_LOWER_BOUND = -11000
S8_UPPER_BOUND = 4800
MASK_THRESH = 4500
S8_INTERVAL = S8_UPPER_BOUND - S8_LOWER_BOUND
NUM_INVALID_LEFT_COLS = 50
NUM_INVALID_RIGHT_COLS = 20

def process_snap_export():
    buffer_path = os.path.join(CREODIAS_PATH, 'data')

    # important channel: confidence_in, S8_BT_in
    # SLSTR subset tif for 4 layers: 0 = confidence_in, 1 = S8_BT_in, 2-3 = lattitude/longitude ?????
    list_files = list(pathlib.Path(buffer_path).rglob('*.tif'))
    if len(list_files) == 0:
        print('No slstr data found')
        exit()
    slstr_arr, filtered_files = collect_data_from_files(list_files)
    product_list = [str(prod.parent) for prod in filtered_files]

    # get position of filled pixels
    filled = slstr_arr[:, 1] == SLSTR_FILL_VALUE

    # normalize data and masks
    normed_arr = np.zeros_like(slstr_arr, dtype='float32')
    normed_data = normalize_data(slstr_arr[:, 1])
    normed_data[filled] = 1
    normed_arr[:, 1] = normed_data
    normed_arr[:, 0] = slstr_arr[:, 0] > MASK_THRESH

    # transpose and resize to same resolution
    resized_tir = resize_to_patchify(normed_arr[:, 1])
    resized_mask = resize_to_patchify(normed_arr[:, 0])
    print(resized_mask.shape, resized_tir.shape)
    slice_data(resized_tir, resized_mask, product_list)


def collect_data_from_files(list_files):
    data_products = []
    filtered_files = []
    for path in list_files:
        # print(path)
        with rasterio.open(path) as tif_file:
            img = tif_file.read()
        # filter inhomogenous sized products
        if abs(img.shape[1] - SLSTR_SHAPE[1]) < 5 and abs(img.shape[2] - SLSTR_SHAPE[0]) < 5:
            if img.shape[0] == 4:
                img = img[0:2]

            # print(np.unique(img[0], return_counts=True))
            # plt.imshow(img[0])
            # plt.show()

            # clipping not neccessary, if fill values are processed, too
            # cleansed = img[:, :, NUM_INVALID_LEFT_COLS:-NUM_INVALID_RIGHT_COLS]

            resized_prod = []
            for layer in img:
                resized_prod.append(cv2.resize(layer, SLSTR_SHAPE, interpolation=cv2.INTER_NEAREST))#INTER_LINEAR_EXACT))
            img = np.array(resized_prod)

            data_products.append(img)
            filtered_files.append(path)
    return np.array(data_products), filtered_files

def resize_to_patchify(data_arr):
    """Resize product to sliceable size (256*X)

    Parameter:
    data_arr: Dataset array
    """
    if len(data_arr.shape) == 2:
        data_arr = np.expand_dims(data_arr, axis=0)
    arr_shape = data_arr.shape
    dim_size_x = round(arr_shape[1] / 256) * 256
    dim_size_y = round(arr_shape[2] / 256) * 256
    resolution = (dim_size_y, dim_size_x)
    resized_arr = []
    for product in range(arr_shape[0]):
        resized_arr.append(cv2.resize(data_arr[product], resolution, interpolation=cv2.INTER_NEAREST))#cv2.INTER_LINEAR_EXACT))#INTER_AREA))
    return np.stack(resized_arr, axis=0)


def normalize_data(data_arr):
    data_arr -= S8_LOWER_BOUND
    normed = data_arr / S8_INTERVAL
    normed[normed < 0.0] = 0.0
    normed[normed > 1.0] = 1.0
    return normed


def save_to_npy(list_files, img_arr, mask_arr):
    data_dir = os.path.join(CREODIAS_PATH, 'train')
    under_indexes = [i for i, ch in enumerate(list_files[0]) if ch == '_']
    for i in range(len(list_files)):
        file_name = os.path.basename(list_files[i])
        basename = file_name[under_indexes[9]+1 : under_indexes[-4]]
        dir_name = os.path.join(data_dir, 'S3B_L1_' + basename)
        if os.path.isdir(dir_name) is False:
            os.mkdir(dir_name)
        np.save(os.path.join(dir_name, 'image.npy'), img_arr[i])
        np.save(os.path.join(dir_name, 'mask.npy'), mask_arr[i])
        #print(list_files[i], str(list_files[i]).replace('original\\buffer', 'original\\processed'))
        #os.replace(list_files[i], str(list_files[i]).replace('original\\buffer', 'original\\processed'))


def show_all_patches(data_arr):
    num_patches = data_arr.shape[0]
    for index in range(1, num_patches+1):
        plt.subplot(1, num_patches, index)
        plt.imshow(data_arr[:, :, index-1], cmap='gray')
        plt.title('Data product\'s ' + str(index) + 'th item')
    plt.show()


def slice_data(slstr_arr, mask_arr, file_list):
    for idx, dir in enumerate(file_list):
        slice_product(dir, slstr_arr[idx], mask_arr[idx])


def slice_product(path, img, mask):
    patched_img = sliding_window(img, win_size=256, stride=128)
    patched_mask = sliding_window(mask, win_size=256, stride=128)
    # plt.subplot(1, 2, 1)
    # plt.imshow(patched_mask[0], cmap='gray')
    # plt.subplot(1, 2, 2)
    # plt.imshow(patched_img[0], cmap='gray')
    # plt.show()
    product_dir = path.replace('original\\data', 'original\\patched')
    print(product_dir)
    if os.path.isdir(product_dir) is False:
        os.mkdir(product_dir)
    for patch_idx in range(1, patched_img.shape[0]+1):
        patch = '0' + str(patch_idx)
        if patch_idx < 10:
            patch = '0' + patch
        patch_dir = os.path.join(product_dir, patch)
        if os.path.isdir(patch_dir) is False:
            os.mkdir(patch_dir)
        np.save(os.path.join(patch_dir, 'image.npy'), patched_img[patch_idx-1])
        np.save(os.path.join(patch_dir, 'mask.npy'), patched_mask[patch_idx-1])


def sliding_window(data, win_size, stride=None):
    if stride == None:
        stride = win_size
    id_y = 0
    width, height = data.shape
    slices = []
    while id_y + win_size <= height:
        id_x = 0
        while id_x + win_size <= width:
            slices.append(data[id_x:id_x+win_size, id_y:id_y+win_size])
            id_x += stride
        id_y += stride
    patched_data = np.stack(slices, axis=0)
    return patched_data


if __name__ == '__main__':
    process_snap_export()
    #slice_data()
