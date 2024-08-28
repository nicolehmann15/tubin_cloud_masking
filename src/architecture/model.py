import os.path

import matplotlib.colors
import numpy as np
import time

from skimage import exposure
from tensorflow import keras
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import seaborn as sns
from keras.callbacks import ModelCheckpoint, EarlyStopping, CSVLogger, TensorBoard, ReduceLROnPlateau, LearningRateScheduler

from . import UNet
from .hyperParameter import get_optimizer, get_loss, f1_score, mIoU, dice_loss, mIoU_loss
from .utils import fuse_one_hot, cloud_amount

BANDS = [3, 2, 1, 10]

class CloudSegmentation(object):
    """A class to work on a remote sensing cloud mask task

    Attributes:
    bands: Available spectral bands
    starting_feature_size: Size of feature map of the first convolution layer in the architecture
    num_cls: Number of classes
    dropout_rate: Ratio of Nodes that will be randomly ignored in every batch
    patch_height: Height of trainable image patches
    patch_width: Width of trainable image patches
    learning_rate: Ratio of how to learn from training predictions
    model: The cloud segmentation neural network
    history: Statistical training history of the model
    """
    def __init__(self, bands, starting_feature_size, num_cls, activation, dropout_rate, patch_height, patch_width=None):
        self.bands = bands
        self.starting_feature_size = starting_feature_size
        self.num_cls = num_cls
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.patch_height = patch_height
        if patch_width is None:
            self.patch_width = patch_height
        else:
            self.patch_width = patch_width
        self.decay_factor = 5
        self.learning_rate = 0.0008
        self.model = None
        self.history = None

    def create_model(self, opt, lr, loss, metrics):
        """Create and Compile the model preparing all model parameter

        Parameter:
        opt: Optimizer in string format
        lr: Learning rate to alter the weights smoothly
        loss: Loss function in string/function pointer format
        metrics: list of metrics in string/function pointer format
        """
        params = {
            'patch_height': self.patch_height,
            'patch_width': self.patch_width,
            'activation': self.activation,
            'L1reg': 0.01,
            'L2reg': 0.01,
            'batch_norm_momentum': 0.99,
            'starting_feature_size': self.starting_feature_size,
            'dropout_rate': self.dropout_rate,
            'dropout_on_last_layer_only': False
        }
        self.model = UNet.u_net_model(params, len(self.bands), self.num_cls)
        self.history = None
        self.learning_rate = lr
        optimizer = get_optimizer(opt, lr)
        self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        # self.model.summary()

    def create_backbone_model(self, unet_instance, opt, lr, loss, metrics, backbone='l8'):
        params = {
            'patch_height': self.patch_height,
            'patch_width': self.patch_width,
            'activation': self.activation,
            'L1reg': 0.01,
            'L2reg': 0.01,
            'batch_norm_momentum': 0.99,
            'starting_feature_size': self.starting_feature_size,
            'dropout_rate': self.dropout_rate,
            'dropout_on_last_layer_only': False
        }
        if backbone == 'l8':
            self.model = UNet.transfer_l8_model(unet_instance, params, len(self.bands), self.num_cls)
        elif backbone == 'resnet':
            self.model = UNet.transfer_resnet_model(params, len(self.bands), self.num_cls)
        self.history = None
        self.learning_rate = lr
        optimizer = get_optimizer(opt, lr)
        self.model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
        self.model.summary()

    def load_model(self, model_path, history_path='', custom_loss=None, compile=True):
        """Load a pre-trained model and belonging history

        Parameter:
        model_path: path to specific model
        history_path: path to model's history
        custom_loss: loss function used in training
        """
        if custom_loss is None or custom_loss == 'binary_crossentropy':
            self.model = keras.models.load_model(model_path, compile=compile, custom_objects={'f1_score': f1_score,
                                                                             'mIoU': mIoU})
        elif custom_loss == 'dice_loss':
            self.model = keras.models.load_model(model_path, compile=compile, custom_objects={'f1_score': f1_score,
                                                                             'mIoU': mIoU,
                                                                             'dice_loss': dice_loss})
        elif custom_loss == 'mIoU_loss':
            self.model = keras.models.load_model(model_path, compile=compile, custom_objects={'f1_score': f1_score,
                                                                             'mIoU': mIoU,
                                                                             'mIoU_loss': mIoU_loss})
        if history_path != '':
            self.history = np.load(history_path, allow_pickle='TRUE').item()

    def train(self, save_path, extension, num_epochs, train_data, val_data, grid_search=False, initial_epoch=0):
        """Train the model and save into save_path

        Parameter:
        save_path: path where to save trained model
        extension: model information -DD_MM-
        num_epochs: number of epochs to train
        train_data: training dataset
        val_data: validation dataset
        """
        self.epochs = num_epochs
        self.decay_rate = self.learning_rate * self.epochs * self.decay_factor
        check_path = './../models/checkpoints/strongest-weights-' + extension + '.hdf5'
        checkpoint_cb = ModelCheckpoint(check_path, monitor='val_f1_score', verbose=1, mode='max', save_best_only=True)
        csv_logger_cb = CSVLogger('./../reports/csv_logger_' + extension + '.log')
        early_stop_cb = EarlyStopping(monitor='val_f1_score', patience=5, verbose=1, mode='max')
        lr_plateau_cb = ReduceLROnPlateau(monitor='val_f1_score', factor=0.8, patience=3, min_delta=0.005, verbose=1, mode='max', min_lr=1e-6)
        lr_decay_cb = LearningRateScheduler(schedule=self.exp_decay, verbose=1)
        # TODO: last step before transfer learning: find best model between lr_plateau_cb, lr_decay_cb and lr grid_search
        callbacks = []
        if grid_search is False:
            callbacks = [checkpoint_cb, csv_logger_cb, lr_plateau_cb] #, early_stop_cb]#, lr_decay_cb]
        trained = self.model.fit(train_data,
                                 epochs=num_epochs,
                                 verbose=1,
                                 validation_data=val_data,
                                 initial_epoch=initial_epoch,
                                 callbacks=callbacks)
        if self.history != None and len(trained.history.keys()) != 0:
            for key in self.history.keys():
                for stat_idx in range(len(trained.history[key])):
                    self.history[key].append(trained.history[key][stat_idx])
        else:
            self.history = trained.history
        np.save(save_path + 'history/' + extension + '.npy', self.history)
        self.model.save(save_path + 'models/' + extension + '.hdf5')

    def draw_history(self):
        """Plot the attached history"""
        if self.history:
            accuracy = np.array(self.history['accuracy'])
            val_accuracy = np.array(self.history['val_accuracy'])
            loss = np.array(self.history['loss'])
            val_loss = np.array(self.history['val_loss'])
            f1score = np.array(self.history['f1_score'])
            val_f1score = np.array(self.history['val_f1_score'])
            miou = np.array(self.history['mIoU'])
            val_miou = np.array(self.history['val_mIoU'])
            num_epochs = len(accuracy)
            epoch_list = np.arange(1, num_epochs + 1)

            plt.subplot(2, 2, 1)
            plt.plot(epoch_list, loss, '-.', label='training')
            plt.plot(epoch_list, val_loss, '-.', label='validation')
            plt.title('Mean loss')
            plt.ylabel('Loss')
            plt.ylim((0, 0.8))
            plt.xlabel('Epochs')
            plt.xlim(1, num_epochs+1)
            plt.xticks(np.arange(1, num_epochs + 1, step=2))
            plt.legend(loc="upper right")
            axes = plt.gca()
            #axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            axes.yaxis.grid()
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axes.xaxis.grid()
            axes.tick_params(axis="y", which='both', direction="in")
            axes.tick_params(axis="x", which='both', direction="in")

            plt.subplot(2, 2, 2)
            plt.plot(epoch_list, accuracy, '-.', label='training')
            plt.plot(epoch_list, val_accuracy, '-.', label='validation')
            plt.title('Mean accuracy')
            plt.ylabel('Accuracy')
            plt.ylim((0.6, 1))
            plt.xlabel('Epochs')
            plt.xlim(1, num_epochs+1)
            plt.xticks(np.arange(1, num_epochs + 1, step=2))
            plt.legend(loc="lower right")
            axes = plt.gca()
            axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            axes.yaxis.grid()
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axes.xaxis.grid()
            axes.tick_params(axis="y", which='both', direction="in")
            axes.tick_params(axis="x", which='both', direction="in")

            plt.subplot(2, 2, 3)
            plt.plot(epoch_list, f1score, '-.', label='training')
            plt.plot(epoch_list, val_f1score, '-.', label='validation')
            plt.title('Mean F1-score')
            plt.ylabel('F1-score')
            plt.ylim((0.6, 1))
            plt.xlabel('Epochs')
            plt.xlim(1, num_epochs+1)
            plt.xticks(np.arange(1, num_epochs + 1, step=2))
            plt.legend(loc="lower right")
            axes = plt.gca()
            axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            axes.yaxis.grid()
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axes.xaxis.grid()
            axes.tick_params(axis="y", which='both', direction="in")
            axes.tick_params(axis="x", which='both', direction="in")

            plt.subplot(2, 2, 4)
            plt.plot(epoch_list, miou, '-.', label='training')
            plt.plot(epoch_list, val_miou, '-.', label='validation')
            plt.title('Mean IoU')
            plt.ylabel('mIoU')
            plt.ylim((0.5, 1))
            plt.xlabel('Epochs')
            plt.xlim(1, num_epochs+1)
            plt.xticks(np.arange(1, num_epochs + 1, step=2))
            plt.legend(loc="lower right")
            axes = plt.gca()
            axes.yaxis.set_minor_locator(ticker.MultipleLocator(0.025))
            axes.yaxis.grid()
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            axes.xaxis.grid()
            axes.tick_params(axis="y", which='both', direction="in")
            axes.tick_params(axis="x", which='minor', direction="in", length=4)
            axes.tick_params(axis="x", which='major', direction="in", length=7)
            plt.tight_layout()
            plt.show()
        else:
            print('There is no history to be plotted.')

    def predict(self, ds):
        """Predict masks of image patches

        Parameter:
        ds: image dataset which can be fed directly into the model
        """
        start = time.time()
        pred_masks = self.model.predict(ds)
        pred_masks = tf.nn.softmax(pred_masks)
        pred_masks = np.argmax(pred_masks, axis=3, keepdims=True)
        #pred_masks = fuse_one_hot(pred_masks)
        end = time.time()
        print(f'The Prediction took {round((end - start) / 60, 1)} minutes')
        return pred_masks

    def evaluate_prediction(self, pred_masks, gt_ds):
        """Evaluate the prediction of masks with ground truth
        Saves conf_matrix, recall, precision, F-score and Jaccard Index in class

        Parameter:
        pred_masks: predicted masks
        gt_ds: ground truth dataset
        """
        gt_images, gt_masks = gt_ds
        #print(str(np.sum(np.logical_and(gt_masks[:, :, :, 0] == 0.0, gt_masks[:, :, :, 1] == 0))) + ' pixel neither have a classification as clear nor as cloudy')
        test_masks = (gt_masks[:, :, :, 1] == 1.0).astype(np.float32)
        print(test_masks.shape, pred_masks.shape)
        patch_size = test_masks[0].shape[0] * test_masks[0].shape[1]
        cloud_amount(test_masks, patch_size)
        if pred_masks.shape == test_masks.shape:
            TP = len(np.where(np.reshape(pred_masks, (-1,)) + np.reshape(test_masks, (-1,)) == 2.0)[0])
            FP = len(np.where(np.reshape(pred_masks, (-1,)) - np.reshape(test_masks, (-1,)) == 1.0)[0])
            TN = len(np.where(np.reshape(pred_masks, (-1,)) + np.reshape(test_masks, (-1,)) == 0.0)[0])
            FN = len(np.where(np.reshape(pred_masks, (-1,)) - np.reshape(test_masks, (-1,)) == -1.0)[0])
            confusion_matrix = np.around((np.array([[TP, FP], [FN, TN]]) / (TP+FP+FN+TN)) * 100, 2)
            print(confusion_matrix)
            # sns.heatmap mit f.e. TP / All (TP + FP + TN + FN)
            ax = sns.heatmap(np.array(confusion_matrix),
                        cmap='rocket_r',
                        vmin=0, vmax=100,
                        annot=True,
                        fmt='.0f',
                        annot_kws={
                            'fontsize': 16
                        },
                        xticklabels=['cloudy ground truth', 'clear ground truth'],
                        yticklabels=['cloudy predicted', 'clear predicted'])
            for text in ax.texts:
                text.set_text(text.get_text() + "%")
            plt.show()
            if TP == 0:
                if TN == 0:
                    self.accuracy = 0.0
                else:
                    self.accuracy = round((TP + TN) / (TP + FP + FN + TN), 4)
                self.recall = 0.0
                self.precision = 0.0
                self.f1_score = 0.0
                self.miou = 0.0
            else:
                self.accuracy = round((TP + TN) / (TP + FP + FN + TN), 4)
                self.recall = round(TP / (TP + FN), 4)
                self.precision = round(TP / (TP + FP), 4)
                self.f1_score = round(2 * (self.recall * self.precision) / (self.recall + self.precision), 4)
                self.miou = round(0.5 * ((TP / (TP + FP + FN)) + (TN / (TN + FP + FN))), 4)

            print(f'Accuracy: {self.accuracy}')
            print(f'Precision: {self.precision}')
            print(f'Recall: {self.recall}')
            print(f'F1-Score: {self.f1_score}')
            print(f'mIoU-Index: {self.miou}')

            #cmap = matplotlib.colors.ListedColormap(['royalblue', 'gold'])
            #t_masks = np.ma.masked_where(test_masks == 0.0, test_masks)
            #p_masks = np.ma.masked_where(pred_masks == 0.0, pred_masks)
            cmap = matplotlib.colors.ListedColormap(['royalblue', 'gold'])
            cmap.set_bad(color='royalblue')
            for i in range(200):
                plt.subplot(1, 3, 1)
                if len(gt_images.shape) == 3:
                    plt.imshow(gt_images[i, :, :, :3], label='gt image', interpolation='nearest')
                    plt.axis('off')
                else:
                    plt.imshow(gt_images[i, :, :, :3], label='gt image', interpolation='nearest')
                    plt.axis('off')
                plt.title('Cloud image ground truth')

                plt.subplot(1, 3, 2)
                plt.imshow(test_masks[i], label='gt mask', cmap=cmap)
                plt.title('Cloud mask ground truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_masks[i], label='pred mask', cmap=cmap)
                plt.title('Predicted cloud mask')
                plt.axis('off')
                plt.tight_layout()
                plt.show()

                CONTRAST_SCALE_FACTOR = 1.2
                plt.subplot(1, 3, 1)
                if len(gt_images.shape) == 3:
                    plt.imshow(gt_images[i, :, :, :3] * CONTRAST_SCALE_FACTOR, label='gt image',
                               interpolation='nearest')
                    plt.axis('off')
                else:
                    plt.imshow(gt_images[i, :, :, :3] * CONTRAST_SCALE_FACTOR, label='gt image',
                               interpolation='nearest')
                    plt.axis('off')
                plt.title('Cloud image ground truth')

                plt.subplot(1, 3, 2)
                plt.imshow(test_masks[i], label='gt mask', cmap=cmap)
                plt.title('Cloud mask ground truth')
                plt.axis('off')

                plt.subplot(1, 3, 3)
                plt.imshow(pred_masks[i], label='pred mask', cmap=cmap)
                plt.title('Predicted cloud mask')
                plt.axis('off')
                plt.tight_layout()
                plt.show()
        else:
            print('The validation masks have the wrong shape')

    def compare_prediction(self, img, mask):
        """Show an image with its predicted mask

        Parameter:
        img: image to be shown
        mask: mask predicted out of the image
        """
        plt.subplot(1, 2, 1)
        plt.imshow(img[:, :, 0], cmap='gray', label='image')
        plt.subplot(1, 2, 2)
        plt.imshow(mask, cmap='gray', label='mask')
        plt.show()

    def exp_decay(self, epoch):
        """Compute the time dependent learning rate decay

        Parameter:
        epoch: the current epoch
        """
        new_lr = self.learning_rate * np.exp(-self.decay_rate * epoch)
        return new_lr


if __name__ == '__main__':
    network = CloudSegmentation([], BANDS, 2)
