import random

import matplotlib.colors
import numpy as np
import tensorflow_datasets as tfds
import time
from tensorflow import keras
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import cv2
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

from . import UNet
from .modelParameter import get_optimizer, get_loss, f1_score, mIoU, exp_decay
from .utils import fuse_one_hot

BANDS = [3, 2, 1, 11]

class CloudSegmentation(object):
    """A class to work on a remote sensing cloud mask task

    Attributes:
    satellite: Name of the satellite who recorded the data
    dataset: satellite data set
    bands: Available spectral bands
    num_cls: Number of classes
    model: the cloud segmentation neural network
    history: statistics on the trained model
    confusion_matrix: performance matrix of the network
    """
    def __init__(self, satellite, dataset, bands, num_cls):
        self.satellite = satellite
        self.dataset = dataset
        self.bands = bands
        self.num_cls = num_cls

        params = {
            'patch_height': 256,
            'patch_width': 256,
            'activation': 'relu',
            'L2reg': 0.1,
            'batch_norm_momentum': 0.99,
            'dropout': 0.05,
            'dropout_on_last_layer_only': False,
            'overlap': 2
        }
        self.model = UNet.u_net_model(params, len(self.bands), self.num_cls)
        self.history = None

    def load_model(self, model_path, history_path):
        """Load a pre-trained model and belonging history

        Parameter:
        model_path: path to specific model
        history_path: path to model's history
        """
        self.model = keras.models.load_model(model_path, custom_objects={ 'f1_score' : f1_score,
                                                                          'mIoU' : mIoU })
        self.history = np.load(history_path, allow_pickle='TRUE').item()

    def compile_model(self, opt, lr, loss, metrics):
        """Compile the model preparing all model parameter

        Parameter:
        opt: Optimizer in string format
        lr: Learning rate to alter the weights smoothly
        loss: Loss function in string format
        """
        optimizer = get_optimizer(opt, lr)
        loss_func = get_loss(loss)
        self.model.compile(loss=loss_func, metrics=metrics, optimizer=optimizer)
        self.model.summary()

    def train(self, save_path, extension, num_epochs, train_ds, val_ds):
        """Train the model in the save_path

        Parameter:
        save_path: path where save trained model
        extension: model information -DD_MM-
        num_epochs: number of epochs to train
        train_ds: training dataset
        val_split: portion of dataset that is used for validation
        """
        check_path = './../models/checkpoints/strongest-weights-' + extension + '-.hdf5' # weights-improvement-epoch-{epoch:02d}
        patience = 4
        checkpoint_cb = ModelCheckpoint(check_path, monitor='val_f1_score', verbose=1, mode='max', save_best_only=True)
        csv_logger_cb = CSVLogger('./../reports/csv_logger' + self.satellite + '_' + extension + '.log')
        early_stop_cb = EarlyStopping(monitor='val_f1_score', patience=patience, min_delta=0.005, verbose=1, mode='max')
        lr_reduce_cb = ReduceLROnPlateau(monitor='val_f1_score', factor=0.4, patience=patience, min_delta=0.005, verbose=1, mode='max', min_lr=1e-6)
        lr_scheduler_cb = LearningRateScheduler(exp_decay)
        callbacks = [checkpoint_cb, csv_logger_cb, lr_reduce_cb]
        trained = self.model.fit(train_ds,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_data=val_ds,
                                 callbacks=callbacks)
        self.history = trained.history
        np.save(save_path + 'history/' + self.satellite + '_' + extension + '.npy', trained.history)
        self.model.save(save_path + 'models/' + self.satellite + '_' + extension + '.hdf5')

    def draw_history(self):
        """Plot the attached history"""
        if self.history:
            accuracy = np.array(self.history['binary_accuracy'])
            val_accuracy = np.array(self.history['val_binary_accuracy'])
            loss = np.array(self.history['loss'])
            val_loss = np.array(self.history['val_loss'])
            f1score = np.array(self.history['f1_score'])
            val_f1score = np.array(self.history['val_f1_score'])
            miou = np.array(self.history['mIoU'])
            val_miou = np.array(self.history['val_mIoU'])
            epochs = np.arange(1, len(accuracy) + 1)

            plt.subplot(2, 2, 1)
            plt.plot(epochs, accuracy, label='training data')
            plt.plot(epochs, val_accuracy, label='validation data')
            plt.title('Mean accuracy of predictions')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc="lower right")
            plt.ylim((0,1))
            axes = plt.gca()
            axes.yaxis.grid()

            plt.subplot(2, 2, 2)
            plt.plot(epochs, loss, label='training data')
            plt.plot(epochs, val_loss, label='validation data')
            plt.title('Mean cost of predictions')
            plt.ylabel('cost')
            plt.xlabel('epoch')
            plt.legend(loc="upper right")
            axes = plt.gca()
            axes.yaxis.grid()

            plt.subplot(2, 2, 3)
            plt.plot(epochs, f1score, label='training data')
            plt.plot(epochs, val_f1score, label='validation data')
            plt.title('F1-Score of predictions')
            plt.ylabel('f1-score')
            plt.xlabel('epoch')
            plt.legend(loc="lower right")
            plt.ylim((0,1))
            axes = plt.gca()
            axes.yaxis.grid()

            plt.subplot(2, 2, 4)
            plt.plot(epochs, miou, label='training data')
            plt.plot(epochs, val_miou, label='validation data')
            plt.title('Jaccard Index of predictions')
            plt.ylabel('mIoU')
            plt.xlabel('epoch')
            plt.legend(loc="lower right")
            plt.ylim((0,1))
            axes = plt.gca()
            axes.yaxis.grid()
            plt.tight_layout()
            plt.show()
        else:
            print('There is no history to be plotted.')

    def predict(self, ds):
        """Predict masks of image patches

        Parameter:
        patches: array of image patches
        """
        threshold = 0.5
        start = time.time()
        pred_masks = self.model.predict(ds) >= threshold
        end = time.time()
        print(f'The Prediction took {round((end - start) / 60, 1)} minutes')
        return pred_masks

    def evaluate_prediction(self, pred_masks, ds):
        """Evaluate the prediction of masks with ground truth
        Saves conf_matrix, recall, precision, F-score and Jaccard Index in class

        Parameter:
        val_masks: ground-truth masks
        pred_masks: predicted masks
        """
        numpy_ds = iter(tfds.as_numpy(ds))
        test_masks = numpy_ds.__next__()[1]
        test_masks = test_masks[np.newaxis, ...]
        for _, mask in numpy_ds:
            test_masks = np.concatenate((test_masks, mask[np.newaxis, ...]), axis=0)
        pred_masks = fuse_one_hot(pred_masks)
        test_masks = fuse_one_hot(test_masks)
        if pred_masks.shape == test_masks.shape:
            TP = len(np.where(pred_masks + test_masks == 2.0)[0])
            FP = len(np.where(pred_masks - test_masks == 1.0)[0])
            TN = len(np.where(pred_masks + test_masks == 0.0)[0])
            FN = len(np.where(pred_masks - test_masks == -1.0)[0])
            print([[TP, FP], [FN, TN]])
            self.accuracy = round((TP + TN) / (TP + FP + FN + TN), 4)
            self.recall = round(TP / (TP + FN), 4)
            self.precision = round(TP / (TP + FP), 4)
            self.f1_score = round(2 * (self.recall * self.precision) / (self.recall + self.precision), 4)
            self.miou = round(TP / (TP + FP + FN), 4)

            print(f'Accuracy: {self.accuracy}')
            print(f'Precision: {self.precision}')
            print(f'Recall: {self.recall}')
            print(f'F1-Score: {self.f1_score}')
            print(f'mIoU-Index: {self.miou}')

            index = random.randint(0, test_masks.shape[0])
            i = 0
            while index+i < test_masks.shape[0] and i < 5:
                plt.subplot(1, 2, 1)
                plt.imshow(test_masks[index+i], label='test mask')
                plt.title('Cloud mask ground truth')

                plt.subplot(1, 2, 2)
                plt.imshow(pred_masks[index+i], label='pred mask')
                plt.title('Predicted cloud mask')
                plt.show()
                i += 1
        else:
            print('The validation masks have the wrong shape')

    def show_prediction(self, test_mask, pred_mask, superposition=False):
        """Shows indexed predictions alongside the original image patch

        Parameter:
        pred_mask: predicted cloud mask
        superposition: The TP classifications are drawn on the original image patch
        """
        height, width, channel = pred_mask.shape

        rcParams['figure.figsize'] = 13, 8
        fig1, ax1 = plt.subplots(1, 2)
        ax1[0].imshow(cv2.cvtColor(test_mask[:, :, 3], cv2.COLOR_BGR2RGB))
        ax1[0].axis('off')
        if superposition:
            pred_superpos = cv2.cvtColor(test_mask[:, :, 3], cv2.COLOR_BGR2RGB)
            for h in range(height):
                for w in range(width):
                    if pred_mask[h, w] == 1:
                        # TODO: customize color
                        pred_superpos[h, w, :] = (255, 110, 0)
            ax1[1].imshow(pred_superpos)
        else:
            # TODO: customize colors
            colors = ['#62a6c2', 'gold']
            ax1[1].imshow(pred_mask, cmap=matplotlib.colors.ListedColormap(colors))
        ax1[1].axis('off')
        fig2, ax2 = plt.subplots(2, 2)
        ax2[0].imshow(cv2.cvtColor(pred_mask[:, :, 3], cv2.COLOR_BGR2RGB))
        ax2[0].axis('off')
        if superposition:
            pred_superpos = cv2.cvtColor(pred_mask[:, :, 3], cv2.COLOR_BGR2RGB)
            for h in range(height):
                for w in range(width):
                    if pred_mask[h, w] == 1:
                        # TODO: customize color
                        pred_superpos[h, w, :] = (255, 110, 0)
            ax2[1].imshow(pred_superpos)
        else:
            # TODO: customize colors
            colors = ['#62a6c2', 'gold']
            ax2[1].imshow(pred_mask, cmap=matplotlib.colors.ListedColormap(colors))
        plt.show()

if __name__ == '__main__':
    network = CloudSegmentation([], BANDS, 2)