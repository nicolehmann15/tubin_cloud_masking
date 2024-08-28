import os
import random

import matplotlib.pyplot as plt

from data.datasets import Dataset
from architecture.model import CloudSegmentation
from architecture.hyperParameter import get_standard_params, dice_loss, f1_score, mIoU

if __name__ == '__main__':
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '100'
    params = get_standard_params()
    # dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], '/Biome_256_Small_pp/test')
    # dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], '/Biome_256_Small_pp_md/test_showcase')
    dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], 'TUBIN_256_pp_md/test')
    # dataset = Dataset([0], params['num_cls'], params['patch_size'], params['patch_size'], 'Sentinel-3/Creodias/test')
    dataset.create_dataset_np(num_samples=1000)
    num_samples = len(dataset.dataset[0])
    print(str(num_samples) + ' samples in the dataset')
    test_ds = dataset.dataset

    unet = CloudSegmentation(params['BANDS'], params['starting_feature_size'], params['num_cls'], params['activation'], params['dropout_rate'], patch_height=params['patch_size'])
    unet.load_model('./../models/TUBIN_256_pp_md_final_oTexas_07_05_17.hdf5', '', 'mIoU_loss', False) #strongest-weights-L8_256_7vs1_22_09_19.hdf5
    print('\n############## testing ##############')
    pred_masks = unet.predict(test_ds[0])
    unet.evaluate_prediction(pred_masks, test_ds)
    # tensorboard?
