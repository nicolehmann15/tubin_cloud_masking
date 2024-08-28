import os
import pathlib
import shutil
import sys
import time
import subprocess

import numpy as np
import pandas as pd
import cv2
import tifffile as tiff
import rasterio
import utm
import matplotlib.pyplot as plt

CAMPAIGN_PATH = 'D:/Clouds/data/TUBIN/Preprocessing/buffer/230828_Detroit'
TUBIN_PATH = 'D:/Clouds/data/TUBIN/Preprocessing/buffer/'
BW_PNG = '01_bw_png'
DEB_FF = '06_deb_ff'
TEMP = '08_temp'
MAT = '10_mat'
REGISTERED = 'Registered'
SCALE_FACTOR = 256 - 1
TIR_SHAPE = (640, 512)
VIS_SHAPE = (3664, 2748)
cloudy_data_path = 'C:/Users/n_leh/Desktop/Masterarbeit/Praxis/labeling/synthetic/cloudy'
clear_data_path = 'C:/Users/n_leh/Desktop/Masterarbeit/Praxis/labeling/synthetic/clear'


def process_dataset(product):
    masks_path = os.path.join(TUBIN_PATH, 'masks')
    mask_list = []
    mask_files = os.listdir(masks_path)
    for file in mask_files:
        mask = np.expand_dims(cv2.cvtColor(cv2.imread(os.path.join(masks_path, file)), cv2.COLOR_BGR2GRAY), axis=2)
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis=0)
    suffix = get_suffix(product)
    list_files = []
    products = 0
    for campaign in os.listdir(TUBIN_PATH):
        campaign_path = os.path.join(TUBIN_PATH, campaign)
        campaign_patched = campaign_path.replace('/buffer', '/patched')
        if os.path.isdir(campaign_patched) is False:
            os.mkdir(campaign_patched)
        campaign_files = list(pathlib.Path(os.path.join(campaign_path, product)).rglob(suffix))
        for idx, product_path in enumerate(campaign_files):
            if products >= mask_list.shape[0]:
                break
            if str(product_path).split('\\')[-1].split('.')[0] != mask_files[products].split('.')[0]:
                continue
            else:
                print(str(product_path).split('\\')[-1].split('.')[0], mask_files[products].split('.')[0])
            product_dir = str(product_path).split('.')[0].replace('\\buffer', '\\patched').replace('\\Registered', '')
            print(product_dir)
            if os.path.isdir(product_dir) is False:
                os.mkdir(product_dir)
            data_arr = collect_data_from_files([product_path], product)

            # standardize the data
            data_arr = standardize_data(data_arr)
            mask_arr = standardize_data(np.expand_dims(mask_list[products], axis=0))
            cond = data_arr[:, :, :, 3] == 0
            mask_arr[cond, 0] = 0
            data_arr[cond, 0] = 0
            data_arr[cond, 1] = 0
            data_arr[cond, 2] = 0

            #plt.subplot(1, 2, 1)
            #plt.imshow(data_arr[0, :, :, :3], label='gt image')
            #plt.axis('off')
            #plt.subplot(1, 2, 2)
            #plt.imshow(mask_arr[0], label='gt mask')
            #plt.axis('off')
            #plt.show()

            # transpose and resize to same resolution
            resized_data = resize_to_patchify(data_arr)
            patched_data = sliding_window(resized_data[0], win_size=256, stride=256)
            hold_list = []
            for patch in range(patched_data.shape[0]):
                if np.any(patched_data[patch] != 0):
                    hold_list.append(patch)
            patched_data = patched_data[np.array(hold_list)]
            resized_mask = resize_to_patchify(mask_arr)
            patched_mask = sliding_window(resized_mask[0], win_size=256, stride=256)
            patched_mask = patched_mask[np.array(hold_list)]
            print(patched_data.shape, patched_mask.shape)

            #plt.subplot(1, 2, 1)
            #plt.imshow(patched_data[0, :, :, :3], label='gt image')
            #plt.axis('off')
            #plt.subplot(1, 2, 2)
            #plt.imshow(patched_mask[0], label='gt mask')
            #plt.axis('off')
            #plt.show()

            for patch_idx in range(1, patched_data.shape[0] + 1):
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
                print(patched_data[patch_idx -1].shape)
                np.save(os.path.join(patch_dir, 'image.npy'), patched_data[patch_idx - 1])
                np.save(os.path.join(patch_dir, 'mask.npy'), patched_mask[patch_idx - 1])
            #processed_product = product_path.replace('/buffer', '/processed').replace
            #shutil.move(product_path, processed_product)

            products += 1
        if products >= mask_list.shape[0]:
            break

def get_suffix(product):
    if product == BW_PNG:
        return '*.png'
    elif product == DEB_FF:
        return '*.png'
    elif product == TEMP:
        return '*.png'
    elif product == REGISTERED:
        return '*.tiff'


def collect_data_from_files(list_files, product):
    data_products = []
    for path in list_files:
        if TEMP == product:
            img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
            # print(np.max(img), np.min(img))
        elif BW_PNG == product:
            print(path)
            img = tiff.imread(path)
        elif DEB_FF == product:
            img = tiff.imread(path)
        elif REGISTERED == product:
            tiff_bands = tiff.imread(path)
            vis_bands = tiff_bands[1]
            tir_band = np.expand_dims(cv2.cvtColor(tiff_bands[0], cv2.COLOR_BGR2GRAY), axis=2)
            tir_band = standardize_data(tir_band)
            img = np.concatenate((vis_bands, tir_band), axis=2)
        data_products.append(img)
    data_products = np.stack(data_products, axis=0)
    return data_products
    tir_img = standardize_data(data_products[4])
    #print(list_files[4], np.min(tir_img), np.max(tir_img))
    #for i in range(0, 21, 1):
    #    print(0.34 + i*0.001)
    #    thresh = np.logical_and(tir_img > 0.0, tir_img < (0.34 + i*0.001))
    #    plt.imshow(thresh, cmap='gray')
    #    plt.show()
    # THRESHOLD 0.312 is best to get best pre-mask for L8
    # THRESHOLD 0.21-0.22 is best to get best pre-mask for TUBIN
    #plt.subplot(2, 2, 1)
    #plt.imshow(tir_img, cmap='gray')
    #thresh = np.logical_and(tir_img > 0.0, tir_img < 0.29)
    #plt.subplot(2, 2, 1)
    #plt.subplot(2, 2, 2)
    #plt.imshow(thresh, cmap='gray')
    #plt.subplot(2, 2, 3)
    #plt.imshow(mask_arr, cmap='gray')
    #plt.show()
    return np.array(data_products)


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
    resized_arr = np.stack(resized_arr, axis=0)
    if len(resized_arr.shape) == 3:
        resized_arr = np.expand_dims(resized_arr, axis=3)
    return resized_arr


def standardize_data(data_arr):
    """Standardize to interval [0.0, 1.0]

        Parameter:
        data_arr: Dataset array
        """
    data_arr = np.round(data_arr / SCALE_FACTOR, 5)
    return data_arr


def save_to_npy(list_files, img_arr, mask_arr):
    data_dir = os.path.join(TUBIN_PATH, 'data')
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
    num_patches = data_arr.shape[2]
    for index in range(1, num_patches+1):
        plt.subplot(1, num_patches, index)
        plt.imshow(data_arr[:, :, index-1], cmap='gray')
        plt.title('Data product\'s ' + str(index) + 'th item')
    plt.show()


def slice_data(slstr_arr, mask_arr):
    data_path = os.path.join(TUBIN_PATH, 'data')
    for idx, dir in enumerate(os.listdir(data_path)):
        dir_path = os.path.join(data_path, dir)
        slice_product(dir_path, slstr_arr[idx], mask_arr[idx])


def slice_product(path, img, mask):
    patched_img = sliding_window(img, win_size=256, stride=128)
    patched_mask = sliding_window(mask, win_size=256, stride=128)
    plt.subplot(1, 3, 1)
    plt.imshow(patched_mask[0], cmap='gray')
    plt.subplot(1, 3, 2)
    plt.imshow(patched_img[0, : , :, 0], cmap='gray')
    mask = np.ones_like(patched_mask)
    mask[np.where(patched_mask < 0.7)] = 0
    plt.subplot(1, 3, 3)
    plt.imshow(mask[0], cmap='gray')
    plt.show()
    exit()
    product_dir = path.replace('original/data', 'original/patched')
    if os.path.isdir(product_dir) is False:
        os.mkdir(product_dir)
    for patch_idx in range(1,patched_img.shape[0]+1):
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
    width, height, _ = data.shape
    slices = []
    while id_y + win_size <= height:
        id_x = 0
        while id_x + win_size <= width:
            slices.append(data[id_x:id_x+win_size, id_y:id_y+win_size])
            id_x += stride
        id_y += stride
    patched_data = np.stack(slices, axis=0)
    return patched_data


def load_csv(path):
    product_list = []
    coord_list = []
    for csv in [file for file in os.listdir(path) if file.endswith('csv')]:
        product_list.append('_'.join(pathlib.Path(csv).stem.split('_')[:-1]))
        # print('_'.join(pathlib.Path(csv).stem.split('_')[:-1]))
        csv_path = os.path.join(path, csv)
        df = pd.read_csv(csv_path)
        # pixels = [(x, y) for x in df['pixel_x'].values for y in df['pixel_y'].values]
        coord = list(zip(df['lat_deg'].values, df['lon_deg'].values))
        coord_list.append(coord)
    return product_list, coord_list


def overlay_bands(path):
    tir_pixels = [(x, y) for y in range(TIR_SHAPE[1])[-1::-1] for x in range(TIR_SHAPE[0])[-1::-1]]
    vis_pixels = [(x, y) for y in range(VIS_SHAPE[1])[-1::-1] for x in range(VIS_SHAPE[0])[-1::-1]]
    for campaign in os.listdir(path):
        campaign_path = os.path.join(path, campaign)
        print(campaign)
        vis_path = os.path.join(campaign_path, BW_PNG) # TODO: change this to DEB_FF
        tir_path = os.path.join(campaign_path, TEMP)
        tir_product_list, tir_coord_list = load_csv(tir_path)
        for idx_t, tir_prod in enumerate([prod for prod in os.listdir(tir_path) if prod.split('.')[0] in tir_product_list and prod.endswith('.png')]):
            print(idx_t, tir_prod)
            tir_img_path = os.path.join(tir_path, tir_prod)
            tir_img = cv2.imread(str(tir_img_path), cv2.IMREAD_GRAYSCALE)
            tir_coords = tir_coord_list[idx_t]
            #print(tir_coords)
            # print(np.max(tir_img), np.min(tir_img))
            for idx_v, vis_prod in enumerate([prod.split('.')[0] for prod in os.listdir(vis_path) if prod.endswith('.png')]):
                print(idx_v, vis_prod)
                vis_tif = rasterio.open(os.path.join(vis_path, vis_prod + '.tif'))
                with tiff.TiffFile(os.path.join(vis_path, vis_prod + '.tif')) as vis_tiff:
                    geo_zone = vis_tiff.geotiff_metadata['GTCitationGeoKey'][-3:]
                    del vis_tiff
                print(vis_tif.bounds) # --> WSG-84 - UTM-Koordinaten
                # print(vis_tif.meta)
                vis_png = rasterio.open(os.path.join(vis_path, vis_prod + '.png'))
                step_x_v = (vis_tif.bounds.right - vis_tif.bounds.left) / VIS_SHAPE[0]
                coord_x = list(np.arange(vis_tif.bounds.right, vis_tif.bounds.left, -step_x_v))
                step_y_v = (vis_tif.bounds.top - vis_tif.bounds.bottom) / VIS_SHAPE[1]
                coord_y = list(np.arange(vis_tif.bounds.bottom, vis_tif.bounds.top, step_y_v))
                # print(step_x_v, step_y_v)
                vis_coord_list = [(x, y) for y in coord_y for x in coord_x]
                #print(vis_coord_list)
                coord_transform = []
                for x, y in vis_pixels:
                    # TODO: duration is far too long --> shorten the computation:
                    ###compute the first three rows and compute the rest using the mean of the step size
                    if x == 3:
                        break
                    #print(x * VIS_SHAPE[1] + y, x, y, len(vis_coord_list))
                    coord_x, coord_y = vis_coord_list[x * VIS_SHAPE[1] + y]
                    print(coord_x, coord_y)
                    vis_coords = utm.to_latlon(coord_x, coord_y, int(geo_zone[:2]), geo_zone[2:])[-1::-1]
                    #print(vis_coords)
                    #print(vis_coords)
                    coord_transform.append(vis_coords)
                step_x_v2 = coord_transform[1][0] - coord_transform[0][0]
                step_y_v2 = coord_transform[1][1] - coord_transform[0][1]
                print(step_x_v2, step_y_v2)
                step_x_t = tir_coords[1][0] - tir_coords[0][0]
                step_y_t = tir_coords[1][1] - tir_coords[0][1]
                print(step_x_t, step_y_t)
                print(f'difference between step_sizes is: {step_x_t / step_x_v2} or {step_y_t / step_y_v2}')
                # factor: 20.9 for x | 3,47 for y
                exit()
            exit()
        exit()


def rotate_vis():
    campaign_list = os.listdir(TUBIN_PATH)
    for campaign in campaign_list:
        campaign_path = os.path.join(TUBIN_PATH, campaign)
        print(campaign_path)
        list_files = list(pathlib.Path(os.path.join(campaign_path, DEB_FF)).rglob(get_suffix(DEB_FF)))
        for path in list_files:
            img = cv2.imread(str(path))
            img = np.rot90(img)
            img = np.rot90(img)
            cv2.imwrite(str(path), img)


def register_vistir(path):
    for campaign in os.listdir(path):
        campaign_path = os.path.join(path, campaign)
        print(campaign_path)
        vis_path = os.path.join(campaign_path, DEB_FF)
        if not os.path.isdir(vis_path):
            print('no vis images available')
            continue
        tir_path = os.path.join(campaign_path, TEMP)
        registered_path = os.path.join(campaign_path, REGISTERED)
        if os.path.isdir(registered_path) is False:
            os.mkdir(registered_path)
        for vis_prod in [prod for prod in os.listdir(vis_path) if prod.endswith('.png')]:
            vis_img_path = os.path.join(vis_path, vis_prod)
            subprocess.call(["python", "C:/Users/n_leh/Desktop/Masterarbeit/Tools/Registration/TUBIN-image-registration/vistir_stitcher.py",
                             "-v", vis_img_path,
                             "-t", tir_path,
                             "-o", registered_path])


def process_registered():
    for campaign in os.listdir(TUBIN_PATH):
        campaign_registered = os.path.join(TUBIN_PATH, campaign + '/Threshed')
        for reg_product in [prod for prod in list(os.listdir(campaign_registered)) if '.tiff' in prod]:
            prod_path = os.path.join(campaign_registered, reg_product)
            print(prod_path)
            # first dimension: TIR / VIS
            # last dimension: RGB - auch für TIR --> IMREAD_GRAYSCALE!
            tiff_bands = tiff.imread(prod_path)
            vis_bands = tiff_bands[1]
            tir_band = np.expand_dims(cv2.cvtColor(tiff_bands[0], cv2.COLOR_BGR2GRAY), axis=2)
            tir_band = standardize_data(tir_band)
            thresh = np.logical_and(tir_band > 0.0, tir_band < 0.218)
            thresh_path = prod_path.split('.')[0] + '.png'
            #cv2.imwrite(thresh_path, thresh * 255)
            #exit()
            for i in range(0, 21, 1):
                print(0.175 + i * 0.01)
                thresh = np.logical_and(tir_band > 0.0, tir_band < (0.175 + i * 0.01))
                plt.subplot(2, 2, 1)
                plt.imshow(tir_band, cmap='gray')
                #plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 2)
                plt.imshow(thresh, cmap='gray')
                #plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.subplot(2, 2, 3)
                plt.imshow(vis_bands)
                #plt.axis('off')
                plt.xticks([])
                plt.yticks([])
                plt.show()
            exit()
            #tiff_bands = np.concatenate((vis_bands, tir_band), axis=2)


# 240208_West_Africa -              / cloudy_23+-1
# 240216_West_Africa - clear_134+-10
# 240228_Texas_Fire  - clear_69+-5 / cloudy_23+-1
# 240229_Mexico_City -              / cloudy_23+-1
# 240229_West_Africa - clear_76..123 / cloudy_23+-1
# 240316_New_Zealand -              / cloudy_24+-1

def create_cloudy_clear_tir():
    #for campaign in os.listdir(TUBIN_PATH):
    campaign_vis_clear = os.path.join(TUBIN_PATH, '240316_New_Zealand' + '/06_deb_ff_clear')
    campaign_vis_cloudy = os.path.join(TUBIN_PATH, '240316_New_Zealand' + '/06_deb_ff_cloudy')
    campaign_tir = os.path.join(TUBIN_PATH, '240316_New_Zealand' + '/08_temp')
    campaign_synthetized = os.path.join(TUBIN_PATH, '240316_New_Zealand' + '/Synthetized')
    if not os.path.isdir(campaign_synthetized):
        os.mkdir(campaign_synthetized)
    #for product in os.listdir(campaign_vis_clear):
    #    clear_path = os.path.join(campaign_vis_clear, product)
    #    clear_tir = np.random.randint(65, 76, (2748, 3664, 1)).astype(np.float32)
    #    timestamp = product.split('_')[2:]
    #    combined_path = os.path.join(campaign_synthetized, 'TUBIN_' + '_'.join(timestamp).split('.')[0] + '.tiff')
    #    clear_vis = cv2.imread(clear_path)
    #    clear_bands = np.concatenate((clear_vis, clear_tir), axis=2)
    #    cv2.imwrite(combined_path, clear_bands)
    for product in os.listdir(campaign_vis_cloudy):
        cloudy_tir = np.random.randint(23, 25, (2748, 3664, 1)).astype(np.float32)
        cloudy_path = os.path.join(campaign_vis_cloudy, product)
        timestamp = product.split('_')[2:]
        combined_path = os.path.join(campaign_synthetized, 'TUBIN_' + '_'.join(timestamp).split('.')[0] + '.tiff')
        cloudy_vis = cv2.imread(cloudy_path)
        cloudy_bands = np.concatenate((cloudy_vis, cloudy_tir), axis=2)
        cv2.imwrite(combined_path, cloudy_bands)
    #tir_list = []
    #for tir_prod in os.listdir(campaign_tir):
    #    print(tir_prod)
    #    tir_path = os.path.join(campaign_tir, tir_prod)
    #    tir_list.append(cv2.imread(tir_path, cv2.IMREAD_GRAYSCALE))
    #    if len(tir_list) == 27:
    #        break
    #tir_array = np.stack(tir_list[23:], axis=0)
    #print(tir_array.shape)
    #print(np.min(tir_array), np.max(tir_array), np.mean(tir_array))
    #for syn in os.listdir(campaign_synthetized):
    #    path = os.path.join(campaign_synthetized, syn)
    #    tiff_bands = tiff.imread(path)
    #    plt.imshow(tiff_bands[:, :, 3], cmap='gray')#tiff_bands[:, :, :3].astype(np.int32))
    #    plt.show()


def create_cloudy_clear_masks():
    tir_path = os.path.join(cloudy_data_path, 'tir')
    vis_path = os.path.join(cloudy_data_path, 'vis')
    mask_path = os.path.join(cloudy_data_path, 'mask')
    for vis_prod in os.listdir(vis_path):
        vis_arr = cv2.imread(os.path.join(vis_path, vis_prod))
        shape = vis_arr.shape
        mask = np.ones((shape[0], shape[1]), dtype=np.float32)
        cv2.imwrite(os.path.join(vis_path, vis_prod).replace('\\vis', '/mask'), mask * 255)
    tir_path = os.path.join(clear_data_path, 'tir')
    vis_path = os.path.join(clear_data_path, 'vis')
    mask_path = os.path.join(clear_data_path, 'mask')
    for vis_prod in os.listdir(vis_path):
        vis_arr = cv2.imread(os.path.join(vis_path, vis_prod))
        shape = vis_arr.shape
        mask = np.zeros((shape[0], shape[1]), dtype=np.float32)
        cv2.imwrite(os.path.join(vis_path, vis_prod).replace('\\vis', '/mask'), mask * 255)


def create_cloudy_clear_dataset():
    # create paths
    tir_path = os.path.join(cloudy_data_path, 'tir')
    vis_path = os.path.join(cloudy_data_path, 'vis')
    mask_path = os.path.join(cloudy_data_path, 'mask')
    ds_path = os.path.join(cloudy_data_path, 'dataset')

    # collect products
    tir_files = [os.path.join(tir_path, f) for f in os.listdir(tir_path)]
    vis_files = [os.path.join(vis_path, f) for f in os.listdir(vis_path)]
    mask_files =[os.path.join(mask_path, f) for f in os.listdir(mask_path)]

    # load data
    for idx in range(len(tir_files)):
        if idx < 4:
            tir_arr = np.expand_dims(tiff.imread(tir_files[idx])[:, :, 2], axis=2)
        else:
            tir_arr = np.expand_dims(tiff.imread(tir_files[idx])[:, :, 3], axis=2)
        vis_arr = cv2.imread(vis_files[idx])
        mask_arr = np.expand_dims(cv2.cvtColor(cv2.imread(mask_files[idx]), cv2.COLOR_BGR2GRAY), axis=2)
        data_arr = np.concatenate((vis_arr, tir_arr), axis=2)

        # standardize the data
        data_arr = standardize_data(data_arr)
        mask_arr = standardize_data(mask_arr)

        # transpose and resize to same resolution
        resized_data = resize_to_patchify(data_arr)
        patched_data = sliding_window(resized_data[0], win_size=256, stride=256)
        resized_mask = resize_to_patchify(mask_arr)
        patched_mask = sliding_window(resized_mask[0], win_size=256, stride=256)

        product_dir = os.path.join(ds_path, os.path.basename(vis_files[idx])).split('.')[0]
        if os.path.isdir(product_dir) is False:
            os.mkdir(product_dir)
        for patch_idx in range(1, patched_data.shape[0] + 1):
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
            np.save(os.path.join(patch_dir, 'image.npy'), patched_data[patch_idx - 1])
            np.save(os.path.join(patch_dir, 'mask.npy'), patched_mask[patch_idx - 1])
        # processed_product = product_path.replace('/buffer', '/processed').replace
        # shutil.move(product_path, processed_product)


if __name__ == '__main__':
    #process_snap_export()
    #slice_data()

    process_dataset(REGISTERED)
    #load_csv()

    #overlay_bands(TUBIN_PATH)

    #rotate_vis()
    #register_vistir(TUBIN_PATH)

    #create_cloudy_clear_tir()
    #create_cloudy_clear_masks()
    #create_cloudy_clear_dataset()

    #process_registered()
