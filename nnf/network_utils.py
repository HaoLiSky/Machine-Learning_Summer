"""

"""
import h5py
import numpy as np
import matplotlib.pyplot as plt
from keras.utils import plot_model
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from nnf.custom_keras_utils import rmse_loss, mean_pred, spread
from nnf.custom_keras_utils import LossHistory, endmask


def predict_energy(inputs_file, model_file, weights_file,
                   key='preprocessed/all'):
    with h5py.File(inputs_file, 'r', libver='latest') as h5f:
        padded = h5f[key][()]
        element_counts_list = np.sum(
                h5f['system'].attrs['s_element_counts'], axis=1)
        energy_list = h5f['system'].attrs['s_energies']
    get_custom_objects().update({'rmse_loss': rmse_loss,
                                 'mean_pred': mean_pred,
                                 'spread': spread,
                                 'LossHistory': LossHistory,
                                 'endmask': endmask})
    model = load_model(model_file)
    model.load_weights(weights_file)

    train_inputs = padded.transpose([1, 0, 2])

    train_inputs = [atom for atom in train_inputs]
    train_inputs.append(
        np.reciprocal(np.asarray(element_counts_list).astype(float)))

    predictions = model.predict(train_inputs)
    return predictions, energy_list


def visualize_model(model_file, output_filename):
    get_custom_objects().update({'rmse_loss'  : rmse_loss,
                                 'mean_pred'  : mean_pred,
                                 'spread'     : spread,
                                 'LossHistory': LossHistory,
                                 'endmask'    : endmask})
    model = load_model(model_file)
    plot_model(model, to_file=output_filename, show_shapes=True, rankdir='LR')


def plot_energy_predictions(predicted, actual, save=False,
                            filename='energy_predictions.png'):
    plt.plot(actual, actual, color='k')
    plt.scatter(actual, predicted, color='r')
    if save:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()