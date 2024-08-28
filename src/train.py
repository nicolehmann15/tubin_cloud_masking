import datetime
import os
import tensorflow as tf
import time
import keras

from data.datasets import Dataset
from data.transformation import augmentate
from architecture.model import CloudSegmentation
from architecture.hyperParameter import f1_score, mIoU, dice_loss, mIoU_loss, get_standard_params

if __name__ == '__main__':
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '100'
    params = get_standard_params()
    # dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], 'Biome_256_pp_md/train')
    dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], 'TUBIN_256_pp_md/train')
    #dataset = Dataset([0], params['num_cls'], params['patch_size'], params['patch_size'], 'Sentinel-3/Creodias/train')
    dataset.create_dataset_tf()
    num_samples = int(dataset.dataset.__len__())
    print(str(num_samples) + ' samples in the dataset\n')
    train_ds, val_ds = dataset.train_val_split(val_split=0.05)

    #val_ds
    #val_dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'], 'Biome_256_Small_pp_md/test')
    #val_dataset.create_dataset_tf()
    #val_ds, _ = val_dataset.train_val_split(val_split=0.0)

    save_path = './../'
    date = datetime.datetime.now()
    ds_extension = 'TUBIN_256_pp_md_final_'
    extension = ds_extension + str(date.strftime('%d')) + '_' + str(date.strftime('%m')) + '_' + str(date.strftime('%H'))
    buffer_size = int(train_ds.__len__())
    batch_size = 22 # 16 / 20
    steps_per_epoch = buffer_size // batch_size + 1
    epochs = 60

    learning_rate = 0.0008
    optimizer = 'adam'
    loss = mIoU_loss # 'binary_crossentropy' # dice_loss #
    metrics = ['accuracy', f1_score, mIoU]
    unet = CloudSegmentation(params['BANDS'], params['starting_feature_size'], params['num_cls'], params['activation'], params['dropout_rate'],
                             patch_height=params['patch_size'])
    unet.create_model(optimizer, learning_rate, loss, metrics)
    # unet.create_backbone_model(unet, optimizer, learning_rate, loss, metrics, 'l8')
    # unet.load_model('./../models/checkpoints/strongest-weights-L8_272_31_08_00-.hdf5', '')

    print('\n############## training ##############')
    train_ds = train_ds.shuffle(buffer_size, reshuffle_each_iteration=True) # round(buffer_size / 2), reshuffle_each_iteration=True)
    train_ds = train_ds.map(augmentate, num_parallel_calls=tf.data.AUTOTUNE)
    train_ds = train_ds.batch(batch_size)
    train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)

    val_ds = val_ds.batch(batch_size)
    start = time.time()
    unet.train(save_path, extension, epochs, train_ds, val_ds)
    end = time.time()
    print(f'The Training took {round((end-start)/60, 1)} minutes for {epochs} epochs')
    unet.draw_history()

    #print('\n############## test ##############')
    #dataset = Dataset(params['BANDS'], params['num_cls'], params['patch_size'], params['patch_size'],
    #                  'D:/Clouds/data/TUBIN/TUBIN_256_pp_md/test')
    #dataset.create_dataset_np(num_samples=500)
    #num_samples = len(dataset.dataset[0])
    #print(str(num_samples) + ' samples in the dataset')
    #test_ds = dataset.dataset
    #pred_masks = unet.predict(test_ds[0])
    #unet.evaluate_prediction(pred_masks, test_ds)
