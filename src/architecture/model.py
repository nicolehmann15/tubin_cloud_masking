import matplotlib.colors
import numpy as np
import tensorflow as tf
from tensorflow import keras
from keras.metrics import Accuracy, CategoricalCrossentropy
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
from matplotlib import rcParams
import cv2

from src.architecture import UNet
from src.architecture.modelParameter import get_optimizer, get_loss

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
        self.model = UNet.u_net_model(params, len(self.bands), num_cls)
        self.history = None
        self.confusion_matrix = None

    def load_model(self, model_path, history_path):
        """Load a pre-trained model and belonging history

        Parameter:
        model_path: path to specific model
        history_path: path to model's history
        """
        self.model = keras.models.load_model(model_path)
        self.history = np.load(history_path, allow_pickle='TRUE').item()

    def compile_model(self, opt, lr, loss):
        """Compile the model preparing all model parameter

        Parameter:
        opt: Optimizer in string format
        lr: Learning rate to alter the weights smoothly
        loss: Loss function in string format
        """
        optimizer = get_optimizer(opt, lr)
        loss_func = get_loss(loss)
        self.model.compile(loss=loss_func, metrics=['accuracy'], optimizer=optimizer)
        self.model.summary()

    def train(self, save_path, num_epochs, num_batches, batch_size, train_ds, val_ds):
        """Train the model in the save_path

        Parameter:
        save_path: path to save trained model
        num_epochs: number of epochs to train
        num_batches: number of batches per epoch
        batch_size: number of patches per batch
        train_ds: training dataset
        val_split: portion of dataset that is used for validation
        """
        trained = self.model.fit(train_ds,
                                 batch_size=batch_size,
                                 epochs=num_epochs,
                                 steps_per_epoch=num_batches,
                                 verbose=2,
                                 validation_data=val_ds)
        self.history = trained.history
        np.save(save_path + '/history/' + self.satellite + '.npy', trained.history)
        self.model.save(save_path + '/models/' + self.satellite + '.hdf5')

    def predict(self, patches):
        """Predict masks of image patches

        Parameter:
        patches: array of image patches
        """
        pred_masks = self.model.predict(patches)
        return pred_masks

    def draw_history(self):
        """Plot the attached history"""
        if self.history:
            accuracy = np.array(self.history['accuracy'])
            val_accuracy = np.array(self.history['val_accuracy'])
            loss = np.array(self.history['loss'])
            val_loss = np.array(self.history['val_loss'])
            # TODO: compute f1 from rsNet and jaccard-index from cloud-net
            fscore = np.array(self.history['loss'])
            val_fscore = np.array(self.history['val_loss'])
            jacc = np.array(self.history['loss'])
            val_jacc = np.array(self.history['val_loss'])
            epochs = np.arange(1, len(accuracy) + 1)

            plt.subplot(2, 2, 1)
            plt.plot(epochs, accuracy, label='training data')
            plt.plot(epochs, val_accuracy, label='validation data')
            plt.title('Mean accuracy of predictions')
            plt.ylabel('accuracy')
            plt.xlabel('epoch')
            plt.legend(loc="lower right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            plt.subplot(2, 2, 2)
            plt.plot(epochs, loss, label='training data')
            plt.plot(epochs, val_loss, label='validation data')
            plt.title('Mean cost of predictions')
            plt.ylabel('cost')
            plt.xlabel('epoch')
            plt.legend(loc="upper right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            plt.subplot(2, 2, 3)
            plt.plot(epochs, fscore, label='training data')
            plt.plot(epochs, val_fscore, label='validation data')
            plt.title('F1-Score of predictions')
            plt.ylabel('f1-score')
            plt.xlabel('epoch')
            plt.legend(loc="upper right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))

            plt.subplot(2, 2, 4)
            plt.plot(epochs, jacc, label='training data')
            plt.plot(epochs, val_jacc, label='validation data')
            plt.title('Jaccard Index of predictions')
            plt.ylabel('jaccard')
            plt.xlabel('epoch')
            plt.legend(loc="upper right")
            axes = plt.gca()
            axes.yaxis.grid()
            axes.xaxis.set_major_locator(ticker.MultipleLocator(5))
            axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
            plt.show()
        else:
            print('There is no history to be plotted.')

    def evaluate_prediction(self, pred_masks, val_masks):
        """Evaluate the prediction of masks with ground truth
        Saves conf_matrix, recall, precision, F-score and Jaccard Index in class

        Parameter:
        val_masks: ground-truth masks
        pred_masks: predicted masks
        """
        if pred_masks.shape == val_masks.shape:
            n_masks, height, width = pred_masks.shape
            self.confusion_matrix = np.zeros((2, 2))
            self.accuracy = np.zeros(n_masks)
            self.recall = np.zeros(n_masks)
            self.precision = np.zeros(n_masks)
            self.fscore = np.zeros(n_masks)
            self.jaccard = np.zeros(n_masks)

            TP = len(np.where(pred_masks + val_masks == 2)[0])
            FP = len(np.where(pred_masks - val_masks == 1)[0])
            TN = len(np.where(pred_masks + val_masks == 0)[0])
            FN = len(np.where(pred_masks - val_masks == -1)[0])
            self.confusion_matrix = [[TP,FP], [FN, TN]]
            self.accuracy = (TP + TN) / (TP + FP + FN + TN)
            self.recall = TP / (TP + FN)
            self.precision = TP / (TP + FP)
            self.fscore = 2 * (self.recall * self.precision) / (self.recall + self.precision)
            self.jaccard = TP / (TP + FP + FN)

        else:
            print('The validation masks have the wrong shape')

    def show_prediction(self, pred_mask, superposition=False):
        """Shows indexed predictions alongside the original image patch

        Parameter:
        pred_mask: predicted cloud mask
        superposition: The TP classifications are drawn on the original image patch
        """
        height, width, channel = pred_mask.shape

        rcParams['figure.figsize'] = 13, 8
        fig, ax = plt.subplots(1, 2)
        ax[0].imshow(cv2.cvtColor(pred_mask[:, :, 3], cv2.COLOR_BGR2RGB))
        ax[0].axis('off')
        if superposition:
            pred_superpos = cv2.cvtColor(pred_mask[:, :, 3], cv2.COLOR_BGR2RGB)
            for h in range(height):
                for w in range(width):
                    if pred_mask[h, w] == 1:
                        # TODO: customize color
                        pred_superpos[h, w, :] = (255, 110, 0)
            ax[1].imshow(pred_superpos)
        else:
            # TODO: customize colors
            colors = ['#62a6c2', 'gold']
            ax[1].imshow(pred_mask, cmap=matplotlib.colors.ListedColormap(colors))
        ax[1].axis('off')
        plt.show()

if __name__ == "__main__":
    network = CloudSegmentation([], BANDS, 2)