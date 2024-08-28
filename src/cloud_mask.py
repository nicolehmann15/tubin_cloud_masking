import sys
import os
import pathlib
from PIL import Image
import numpy as np

from architecture.model import CloudSegmentation
from architecture.hyperParameter import get_standard_params, f1_score, mIoU
from architecture.utils import cloud_amount, fuse_one_hot
from data.transformation import rescaling
import data.preproc_l8 as pp_l8

BANDS = [2, 1, 0, 3]

def load_predict_files(input_path, format):
    if os.path.isdir(input_path):
        image_paths = list(pathlib.Path(input_path).rglob('*'))
    else:
        image_paths = [input_path]

    print(image_paths)
    if format == 'png':
        images = []
        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)
        tir_img = images[0]
        vis_img = images[1]
        tir_res = tir_img.shape[:-1]
        vis_res = vis_img.shape[:-1]
        #plt.subplot(1, 3, 1)
        #plt.imshow(tir_img[:, :, 0])
        #plt.subplot(1, 3, 2)
        #plt.imshow(tir_img[:, :, 1])
        #plt.subplot(1, 3, 3)
        #plt.imshow(tir_img[:, :, 2])
        #plt.show()
        #exit()
        #rescaling(tir_img, (4 * vis_res[0], 4 * vis_res[1]))
        vis_img = rescaling(vis_img, tir_res, image_paths[0])
        print(vis_img.shape, tir_img.shape)
        # TODO: welcher Channel von tir??
        image_arr = np.stack([vis_img, tir_img], axis=2)
        print(image_arr.shape)
        exit()
    elif format == 'tif':
        images = []
        for image_path in image_paths:
            img = np.array(Image.open(image_path))
            images.append(img)
        image_arr = np.stack(images, axis=2)
        print(image_arr.shape)
        exit()
    else:
        image_arr = np.array([])
    return image_arr, image_paths

if __name__ == '__main__':
    if len(sys.argv) > 1:
        input_path = str(sys.argv[1])
        if len(sys.argv) > 2:
            format = str(sys.argv[2])
            if len(sys.path) > 3:
                save_path = str(sys.argv[3])
            else:
                print('No save path given')
                exit()
        else:
            print('No data format given, please add a format like png')
            exit()
    else:
        print('No path given!')
        input_path = 'D:/Clouds/data/test_predict/buffer'
        format = 'png'
    params = get_standard_params()
    unet = CloudSegmentation(params['BANDS'], params['starting_feature_size'], params['num_cls'], params['dropout_rate'], params['patch_size'])
    unet.load_model('./../models/checkpoints/strongest-weights-L8_256_13_11_21.hdf5', '')
    predict_data, origin_dict = pp_l8.preprocess_data(input_path, 'predict')
    input_shape = predict_data[0].shape
    print(input_shape)
    print('\n############## predict ##############')
    pred_masks = unet.predict(predict_data)
    pred_masks = fuse_one_hot(pred_masks)
    #cloud_amount(pred_masks, pred_masks.shape[1] * pred_masks.shape[2])
    unet.compare_prediction(predict_data[518], pred_masks[518])

    # save cloud_masks into file
    if origin_dict['og_size'][0] != 256:
        pred_masks = pp_l8.join_patches(origin_dict, pred_masks)
    result_dir = input_path.replace('buffer', 'results')
    if os.path.isdir(result_dir) is False:
        os.mkdir(result_dir)
    for product_idx, dir_path in enumerate(origin_dict['path']):
        mask_name = os.path.basename(dir_path)
        save_path = os.path.join(result_dir, mask_name + '.png')
        png = Image.fromarray(pred_masks[product_idx]).convert("L")
        png.save(save_path) # png-file
        #np.save(save_path, pred_masks) npy-file