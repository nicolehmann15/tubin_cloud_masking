import datetime
import os

from data.datasets import Dataset
from architecture.model import CloudSegmentation
from architecture.modelParameter import f1_score, mIoU

BANDS = [3, 2, 1, 11]
if __name__ == '__main__':
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '100'
    dataset = Dataset('ccava', BANDS, 2, 256, 256, 'D:/Clouds/data/LandSat8/CCAVA_256/test')
    dataset.create_dataset()
    num_samples = int(dataset.dataset.__len__())
    print(str(num_samples) + ' samples in the dataset')
    test_ds = dataset.dataset

    batch_size = 16
    unet = CloudSegmentation('landsat8', dataset, BANDS, 2)
    # re-train a model
    unet.load_model('./../models/landsat8_07_07_10.hdf5', './../history/landsat8_07_07_10.npy')

    print('\n############## testing ##############')
    if num_samples > 500:
        take_it = 500
    else:
        take_it = num_samples
    test_ds = test_ds.take(take_it) # .shuffle(num_samples)
    input = test_ds.batch(batch_size)
    pred_masks = unet.predict(input)
    unet.evaluate_prediction(pred_masks, test_ds)
    # tensorboard?