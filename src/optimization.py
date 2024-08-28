import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from matplotlib import ticker
import matplotlib.colors as mcolors

from data.datasets import Dataset
from architecture.model import CloudSegmentation
from architecture.hyperParameter import f1_score, mIoU, dice_loss, mIoU_loss, get_standard_params

BANDS = [3, 2, 1, 10]
GRID_PATH = './../grid_search/'
COLOR_LIST = ['red', 'sandybrown', 'steelblue', 'lightseagreen', 'darkcyan', 'deepskyblue',
              'orange', 'burlywood', 'goldenrod', 'khaki', 'olive', 'yellow', 'yellowgreen',
              'green', 'darkseagreen', 'aquamarine', 'black', 'grey', 'rosybrown', 'darkred',
              'navy', 'blue', 'slateblue', 'indigo', 'darkviolet', 'violet', 'chocolate',
              'purple', 'magenta', 'hotpink', 'crimson', 'pink', 'beige', 'plum',
              'royalblue', 'mediumturquoise']
NEW_COLOR_LIST = list(mcolors.CSS4_COLORS.keys())[15:]

def get_param_grid(optimization_plan='A'):
    """Return the grid of interesting hyperparameters"""
    starting_feature_size = [12, 16]
    learning_rate = [0.0008, 0.0007, 0.0006, 0.0005, 0.0004, 0.0003, 0.0002, 0.0001]
    dropout_rate = [0.1, 0.2] # 0.15,
    optimizer = ['sgd', 'adam', 'adadelta']
    loss = ['binary_crossentropy', mIoU_loss] #,dice_loss
    activation = ['relu', 'leaky_relu']
    batch_size = [16, 24]
    if optimization_plan == 'A':
        return dict(starting_feature_size=starting_feature_size,
                    dropout_rate=dropout_rate,
                    optimizer=optimizer,
                    loss=loss,
                    activation=activation,
                    batch_size=batch_size)
    else:
        return dict(learning_rate=learning_rate)


def create_model(starting_feature_size=12, dropout_rate=0.2, optimizer='adam', loss=mIoU_loss, activation='leaky_relu', batch_size=20, learning_rate_param=None):
    """Create the unet model considering the given hyperparameters

    Parameter:
    starting_feature_size: Size of feature map of the first convolution layer of the architecture
    dropout_rate: Ratio of Nodes that are ignored in a batch
    optimizer: Optimizer for training as a string
    loss: Loss function for training as a string / function pointer
    """
    print(starting_feature_size, dropout_rate, optimizer, loss, activation, batch_size)
    learning_rate = 0.0005
    if learning_rate_param != None:
        learning_rate = learning_rate_param
    metrics = ['accuracy', f1_score, mIoU]
    num_cls = 2
    unet = CloudSegmentation(BANDS, starting_feature_size, num_cls, activation, dropout_rate, patch_height=256)
    unet.create_model(optimizer, learning_rate, loss, metrics)
    return unet


def custom_h_gridsearch(optimization_plan='A', del_factor=3):
    """The search strategy starts evaluating all candidates with
        a small amount of resources and iteratively selects the best candidates,
        using more and more resources

    Three epochs per training iteration - afterwards compare all stats and
        delete all models except the best 1/3 of all candidates

    Parameter:
    optimization_plan: Which combination of hyperparameters will be investigated
    del_factor: The factor to split good from bad candidates"""
    param_grid = get_param_grid(optimization_plan=optimization_plan)
    num_candidates = np.array([len(param_grid[key]) for key in list(param_grid.keys())]).prod(axis=0)
    print('There are ' + str(num_candidates) + ' candidates available at the start of successive halving\n')
    dataset = Dataset(BANDS, 2, 256, 256, 'D:/Clouds/data/LandSat8/Biome_256_pp_md/train')
    param_candidates = []
    model_candidates = []
    count = 0
    batch_size = 24
    if optimization_plan == 'A':
        epochs_per_iteration = 3
        for starting_feature_size in param_grid['starting_feature_size']:
            for dropout_rate in param_grid['dropout_rate']:
                for optimizer in param_grid['optimizer']:
                    for loss in param_grid['loss']:
                        for activation in param_grid['activation']:
                            for batch_size in param_grid['batch_size']:
                                param_candidates.append((starting_feature_size, dropout_rate, optimizer, loss, activation, batch_size))
                                model_candidates.append(count)
                                count += 1
    elif optimization_plan == 'B':
        epochs_per_iteration = 5
        for learning_rate in param_grid['learning_rate']:
            param_candidates.append((learning_rate,))
            model_candidates.append(count)
            count += 1
    k_cross = 2
    iteration = 0
    while len(model_candidates) > (del_factor - 1):
        print(str(len(model_candidates)) + ' candidates are still available')
        num_samples = 2400 * (iteration+1) #3000 * (iteration+1)
        dataset.create_dataset_tf(num_samples)
        cross_validation_stats = np.zeros((len(model_candidates), k_cross))
        for cross_idx in range(k_cross):
            train_ds, val_ds = dataset.get_kfold_set(k_cross, cross_idx)
            # train_ds.shuffle(num_samples, reshuffle_each_iteration=True)
            train_ds = train_ds.batch(batch_size)# param_candidates[candidate_idx][5])
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.batch(batch_size)# param_candidates[candidate_idx][5])
            for list_idx, candidate_idx in enumerate(model_candidates):
                extension = 'GRID_CANDIDATE_' + str(candidate_idx + 1) + '_iter_' + str(iteration + 1)
                if optimization_plan == 'A':
                    unet = create_model(param_candidates[candidate_idx][0],
                                        param_candidates[candidate_idx][1],
                                        param_candidates[candidate_idx][2],
                                        param_candidates[candidate_idx][3],
                                        param_candidates[candidate_idx][4])
                elif optimization_plan == 'B':
                    unet = create_model(learning_rate_param=param_candidates[candidate_idx][0])
                print('############## ' + str(cross_idx+1) + '.cross training for the ' + str(candidate_idx+1) + '.candidate started ##############')
                unet.train(GRID_PATH, extension, epochs_per_iteration, train_ds, val_ds, grid_search=True)
                candidate_stat = short_evaluation(unet.history)[2]
                cross_validation_stats[list_idx, cross_idx] = candidate_stat
                print('############## ' + str(cross_idx+1) + '.cross training for the ' + str(candidate_idx+1) + '.candidate completed ##############\n')
                # TODO: how to free memory after training
                tf.keras.backend.clear_session()
        stats = list(np.array(cross_validation_stats).mean(axis=1))
        iteration += 1
        score_per_candidate = [(model_candidates[idx], crossed_score) for idx, crossed_score in enumerate(stats)]
        score_per_candidate.sort(key=lambda a: a[1])
        print(score_per_candidate)
        take_part = len(score_per_candidate) - round(len(score_per_candidate) / del_factor)
        del_candidates = list(np.array(score_per_candidate)[:take_part])
        for candidate in del_candidates:
            model_candidates.remove(candidate[0])
    print('The best hyperparameter combination(s) consist(s) of: ', end='')
    for idx in model_candidates:
        print(param_candidates[idx])
    show_histories(param_candidates, num_epochs=iteration * epochs_per_iteration)


def custom_h_gridsearch_show(optimization_plan='A', del_factor=3):
    """The search strategy starts evaluating all candidates with
        a small amount of resources and iteratively selects the best candidates,
        using more and more resources

    Three epochs per training iteration - afterwards compare all stats and
        delete all models except the best 1/3 of all candidates

    Parameter:
    optimization_plan: Which combination of hyperparameters will be investigated
    del_factor: The factor to split good from bad candidates"""
    param_grid = get_param_grid(optimization_plan=optimization_plan)
    num_candidates = np.array([len(param_grid[key]) for key in list(param_grid.keys())]).prod(axis=0)
    print('There are ' + str(num_candidates) + ' candidates available at the start of successive halving\n')
    dataset = Dataset(BANDS, 2, 256, 256, 'D:/Clouds/data/LandSat8/Biome_256_pp_md/train')

    param_candidates = []
    model_candidates = []
    count = 0
    batch_size = 24
    if optimization_plan == 'A':
        epochs_per_iteration = 3
        for starting_feature_size in param_grid['starting_feature_size']:
            for dropout_rate in param_grid['dropout_rate']:
                for optimizer in param_grid['optimizer']:
                    for loss in param_grid['loss']:
                        for activation in param_grid['activation']:
                            for batch_size in param_grid['batch_size']:
                                param_candidates.append((starting_feature_size, dropout_rate, optimizer, loss, activation, batch_size))
                                model_candidates.append(count)
                                count += 1
    elif optimization_plan == 'B':
        epochs_per_iteration = 5
        for learning_rate in param_grid['learning_rate']:
            param_candidates.append((learning_rate,))
            model_candidates.append(count)
            count += 1
    iteration = 0
    while len(model_candidates) > (del_factor - 1):
        print(str(len(model_candidates)) + ' candidates are still available')
        num_samples = 2400 * (iteration+1) # 1000 * (iteration+1)
        dataset.create_dataset_tf(num_samples)
        stats = []
        for candidate_idx in model_candidates:
            train_ds, val_ds = dataset.train_val_split(val_split=0.1)
            train_ds = train_ds.batch(batch_size)# param_candidates[candidate_idx][5])
            train_ds = train_ds.prefetch(buffer_size=tf.data.AUTOTUNE)
            val_ds = val_ds.batch(batch_size)# param_candidates[candidate_idx][5])
            extension = 'GRID_CANDIDATE_' + str(candidate_idx + 1)
            if iteration == 0:
                if optimization_plan == 'A':
                    unet = create_model(param_candidates[candidate_idx][0],
                                        param_candidates[candidate_idx][1],
                                        param_candidates[candidate_idx][2],
                                        param_candidates[candidate_idx][3],
                                        param_candidates[candidate_idx][4],
                                        param_candidates[candidate_idx][5])
                elif optimization_plan == 'B':
                    unet = create_model(learning_rate_param=param_candidates[candidate_idx][0])
            else:
                model_path = os.path.join(os.path.join(GRID_PATH, 'models'), extension + '.hdf5')
                history_path = os.path.join(os.path.join(GRID_PATH, 'history'), extension + '.npy')
                if optimization_plan == 'A':
                    print(param_candidates[candidate_idx][0], param_candidates[candidate_idx][1],
                        param_candidates[candidate_idx][2], param_candidates[candidate_idx][3],
                        param_candidates[candidate_idx][4], param_candidates[candidate_idx][5])
                    unet.load_model(model_path, history_path, custom_loss=param_candidates[candidate_idx][3])
                elif optimization_plan == 'B':
                    print(param_candidates[candidate_idx][0])
                    unet.load_model(model_path, history_path, custom_loss=mIoU_loss)
            print('############## training for the ' + str(candidate_idx+1) + '.candidate started ##############')
            unet.train(GRID_PATH, extension, epochs_per_iteration*(iteration+1), train_ds, val_ds, grid_search=True, initial_epoch=(iteration*epochs_per_iteration))
            candidate_stat = short_evaluation(unet.history)[2]
            print('############## training for the ' + str(candidate_idx+1) + '.candidate completed ##############\n')
            stats.append(candidate_stat)
            #TODO: how to free memory after training
            tf.keras.backend.clear_session()
        iteration += 1
        score_per_candidate = [(model_candidates[idx], score) for idx, score in enumerate(stats)]
        score_per_candidate.sort(key=lambda a: a[1])
        print(score_per_candidate)
        take_part = len(score_per_candidate) - round(len(score_per_candidate) / del_factor)
        del_candidates = list(np.array(score_per_candidate)[:take_part])
        # score_per_candidate = list(np.array(score_per_candidate)[take_part:])
        for candidate in del_candidates:
            model_candidates.remove(candidate[0])
    print('The best hyperparameter combination(s) consist(s) of: ', end='')
    for idx in model_candidates:
        print(param_candidates[idx])
    show_histories(param_candidates, num_epochs=iteration * epochs_per_iteration)


def short_evaluation(history):
    """Extract the core statistics of the training history of a model

    Parameter:
    history: Statistical training history of a model
    """
    val_accuracy = np.array(history['val_accuracy'])
    val_loss = np.array(history['val_loss'])
    val_f1score = np.array(history['val_f1_score'])
    val_miou = np.array(history['val_mIoU'])
    return [val_accuracy[-1], val_loss[-1], val_f1score[-1], val_miou[-1]]


def show_histories(param_candidates, num_epochs):
    # TODO: use _iter_x.npy files and compute mean of them
    extension = 'GRID_CANDIDATE_'
    history_dir = os.path.join(GRID_PATH, 'history')
    for idx, _ in enumerate(param_candidates):
        candidate_ext = extension + str(idx+1) + '.npy'
        history_path = os.path.join(history_dir, candidate_ext)
        history = np.load(history_path, allow_pickle='TRUE').item()
        val_f1score = np.array(history['val_f1_score'])
        x_range = range(1, len(val_f1score)+1)
        if len(val_f1score) == num_epochs:
            plt.plot(x_range, val_f1score, color=NEW_COLOR_LIST[idx], label=candidate_ext)
        else:
            plt.plot(x_range, val_f1score, color=NEW_COLOR_LIST[idx])
    plt.title('Halving Grid Search results')
    plt.ylabel('Validation F1-Score')
    plt.ylim((0.3, 1.0))
    plt.xlabel('Epochs')
    plt.xlim(1, num_epochs)
    plt.xticks(np.arange(1, num_epochs + 1, step=1))
    plt.legend(loc="lower right")
    axes = plt.gca()
    axes.yaxis.grid()
    axes.xaxis.set_minor_locator(ticker.MultipleLocator(1))
    axes.xaxis.grid()
    axes.tick_params(axis="y", which='both', direction="in")
    axes.tick_params(axis="x", which='both', direction="in")
    plt.show()

# Für die Anschaulichkeit: statt immer neue Modelle zu initialisieren, werden die Modelle weitertrainiert
#  somit werden die besten Modelle bis zu 15 Epochen trainiert
#    Grund dafür ist, dass man die history von allen Modellen sehr gut in einen Plot packen kann
if __name__ == '__main__':
    os.environ['TF_DEVICE_MIN_SYS_MEMORY_IN_MB'] = '100'
    #custom_h_gridsearch_show(optimization_plan='A', del_factor=3)
    custom_h_gridsearch(optimization_plan='B', del_factor=2)
    #custom_h_gridsearch_show(optimization_plan='B', del_factor=2)

    #param_candidates = list(range(10))
    #num_epochs = 15
    #show_histories(param_candidates, num_epochs)