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

TIR_SHAPE = (640, 512)
VIS_SHAPE = (3664, 2748)
TUBIN_PATH = 'D:/Clouds/data/TUBIN/UNet_Studentengruppe/tubin_data_U_Net/'

def process_dataset():
    images_path = os.path.join(TUBIN_PATH, 'train_images')
    masks_path = os.path.join(TUBIN_PATH, 'train_masks')
    patch_dir = os.path.join(TUBIN_PATH, 'data')
    img_file_list = list(os.listdir(images_path))
    mask_file_list = [os.path.join(masks_path, f) for f in os.listdir(masks_path) if f.split('.')[0] + '.png' in img_file_list]
    img_file_list = [os.path.join(images_path, f) for f in img_file_list]
    mask_list = []
    for file in mask_file_list:
        mask = np.expand_dims(cv2.cvtColor(cv2.imread(file), cv2.COLOR_BGR2GRAY), axis=2)
        mask_list.append(mask)
    mask_list = np.stack(mask_list, axis=0)
    for idx, img in enumerate(img_file_list):
        patched_path = img.replace('/train_images', '/data').split('.')[0]
        img_arr = cv2.imread(img)
        if os.path.isdir(patched_path) is False:
            os.mkdir(patched_path)

        # standardize the data
        data_arr = standardize_data(img_arr)
        mask_arr = standardize_data(mask_list[idx])

        # transpose and resize to same resolution
        patched_data = sliding_window(data_arr, win_size=256, stride=256)
        patched_mask = sliding_window(mask_arr, win_size=256, stride=256)
        for patch_idx in range(1, patched_data.shape[0] + 1):
            patch = '00' + str(patch_idx)
            if patch_idx < 10:
                patch = '0' + patch
            elif patch_idx >= 100:
                if patch_idx >= 1000:
                    patch = patch[2:]
                else:
                    patch = patch[1:]
            patch_dir = os.path.join(patched_path, patch)
            if os.path.isdir(patch_dir) is False:
                os.mkdir(patch_dir)
            np.save(os.path.join(patch_dir, 'image.npy'), patched_data[patch_idx - 1])
            np.save(os.path.join(patch_dir, 'mask.npy'), patched_mask[patch_idx - 1])


def standardize_data(data_arr):
    """Standardize to interval [0.0, 1.0]

        Parameter:
        data_arr: Dataset array
        """
    data_arr = np.round(data_arr / 255, 5)
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

if __name__ == '__main__':
    process_dataset()
