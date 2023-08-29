import pathlib
import numpy as np
import tensorflow as tf

from .utils import printProgressBar
#from .datasets import Dataset
#data_handling = Dataset('', 0, 2, 0, 0, '')
THRESH = 0.01

#def clean_dataset(dataset, path):
#    home = pathlib.Path(path)
#    path_list = list(home.rglob('**/mask.npy'))
#    cloudless = 0
#    num_samples = len(path_list)
#    i = 0
#    printProgressBar(0, total=num_samples, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='>')
#    for file_path in path_list:
#        data = data_handling.get_cloud_mask_from_file(file_path)
#        cloud_amount = get_cloud_amount_single(data[:, :, 1])
#        if cloud_amount < THRESH:
#            cloudless += 1
#            # remove the tile
#        # Update Progress Bar
#        printProgressBar(i + 1, num_samples, prefix='Progress:', suffix='Complete', decimals=2, length=50, fill='>')
#        i += 1
#    print(str(cloudless) + ' cloudless patches')

def filter_cloudless(img, mask):
    #width, height, _ = mask.shape
    #num_pixels = width * height
    #eq = tf.equal(mask, 1.0)
    #y, _, counts = tf.unique_with_counts(eq, tf.dtypes.int32)
    #print(tf.make_ndarray(counts))
    #cloud_amount = round(counts[1] / num_pixels, 3)
    #print(cloud_amount)
    cloud_amount = get_cloud_amount_single(mask[:, :, 1])
    print(cloud_amount)
    if cloud_amount < THRESH:
        return False
    else:
        return True

def get_cloud_amount_single(data_arr):
    width, height = data_arr.shape
    num_pixels = width * height
    cloudy = np.count_nonzero(data_arr == 1.0)
    cloud_amount = round(cloudy / num_pixels, 3)
    return cloud_amount

if __name__ == '__main__':
    dataset = 'ccava'
    path = 'D:\\Clouds\\data\\Landsat8\\CCAVA_256'
    # clean_dataset(dataset, path)