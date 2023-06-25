import tensorflow as tf
from keras import losses
from src.data.datasets import Dataset
from src.architecture.model import CloudSegmentation

BANDS = [3, 2, 1, 11]

if __name__ == '__main__':
    dataset = Dataset('ccava', BANDS, 2, 256, 256, 'D:/Clouds/data/LandSat8/CCAVA_Small')
    dataset.create_dataset()
    num_samples = int(dataset.dataset.__len__())
    print(str(num_samples) + " samples in the dataset")
    train_ds, val_ds, test_ds = dataset.train_test_split(test_split=0.1, val_split=0.05)

    save_path = './../'
    batch_size = 64
    steps_per_epoch = int(train_ds.__len__()) // batch_size
    epochs = 2
    learning_rate = 0.005

    optimizer = 'adam'
    loss = 'binary_crossentropy' # 'categorical_crossentropy'  # jacc_coef

    unet = CloudSegmentation('landsat8', dataset, BANDS, 2)
    train_ds = train_ds.batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_ds = val_ds.batch(batch_size)
    unet.compile_model(optimizer, learning_rate, loss)
    unet.train(save_path, epochs, steps_per_epoch, batch_size, train_ds, val_ds)


