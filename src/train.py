import datetime
import os
import tensorflow as tf
import time

from data.datasets import Dataset
from data.transformation import augmentate
from architecture.model import CloudSegmentation
from architecture.hyperParameter import f1_score, mIoU, dice_loss, mIoU_loss

BANDS = [3, 2, 1, 11]
if __name__ == '__main__':
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '100'
    dataset = Dataset('biome', BANDS, 2, 224, 224, 'D:/Clouds/data/LandSat8/Biome_224/train')
    dataset.create_dataset()
    num_samples = int(dataset.dataset.__len__())
    print(str(num_samples) + ' samples in the dataset')
    train_ds, val_ds = dataset.train_val_split(val_split=0.05)

    save_path = './../'
    date = datetime.datetime.now()
    ds_extension = 'L8_224_'
    extension = ds_extension + str(date.strftime('%d')) + '_' + str(date.strftime('%m')) + '_' + str(date.strftime('%H'))
    buffer_size = int(train_ds.__len__())
    batch_size = 24
    steps_per_epoch = buffer_size // batch_size + 1
    epochs = 20

    learning_rate = 0.0007
    optimizer = 'adam'
    loss = 'binary_crossentropy' #dice_loss # mIoU_loss # 'binary_crossentropy' #
    metrics = ['accuracy', f1_score, mIoU] # 'binary_accuracy'
    unet = CloudSegmentation('landsat8', dataset, BANDS, num_cls=2)
    # re-train a model
    #unet.load_model('./../models/landsat8_06_07_09.hdf5', './../history/landsat8_06_07_09.npy')
    unet.compile_model(optimizer, learning_rate, loss, metrics)

    print('\n############## training ##############')
    train_ds = train_ds.batch(batch_size)
    #train_ds = train_ds.map(augmentate)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
    #train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE) # .shuffle(batch_size, reshuffle_each_iteration=True) after batch()

    val_ds = val_ds.batch(batch_size)
    start = time.time()
    unet.train(save_path, extension, epochs, train_ds, val_ds)
    end = time.time()
    print(f'The Training took {round((end-start)/60, 1)} minutes for {epochs} epochs')
    unet.draw_history()
