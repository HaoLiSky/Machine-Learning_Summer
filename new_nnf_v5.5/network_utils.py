"""

"""
import os
import h5py
import numpy as np
#import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
from keras.utils import plot_model
from keras.models import load_model
from keras.utils.generic_utils import get_custom_objects
from nnf.custom_keras_utils import rmse_loss, mean_pred, spread
from nnf.custom_keras_utils import LossTracking, endmask
from nnf.batch_preprocess import read_partitions, PartitionProcessor

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# arbitrary fixed seed for reproducibility
np.random.seed(8)


class ModelEvaluator:
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)

    def model_from_file(self):
        self.model_file = self.settings['model_file']
        get_custom_objects().update({'rmse_loss'  : rmse_loss,
                                     'mean_pred'  : mean_pred,
                                     'spread'     : spread,
                                     'LossTracking': LossTracking,
                                     'endmask'    : endmask})
        self.model = load_model(self.model_file)

        if os.path.isfile('ignore_tags'):
            with open('ignore_tags', 'r') as file_:
                ignore_tags = file_.read().split('\n')
                self.ignore_tags = list(filter(None, ignore_tags))
        else:
            self.ignore_tags = []

    def plot_subsample_predictions(self, filename, weights_filename):
        libver = self.settings['libver']
        k_test = self.settings['validation_ind']
        partitions_file = self.settings['partitions_file']
        part_dict = read_partitions(partitions_file)

        with h5py.File(filename, 'r', libver=libver) as h5f:
            self.all_data = h5f['preprocessed/all'][()]
            self.sizes = np.sum(h5f['preprocessed/element_counts'][()], axis=1)
            self.energy_list = h5f['preprocessed/energies'][()]
            self.m_names = [name.decode('utf-8')
                            for name in h5f['preprocessed/m_names'][()]]
        predictions = []

        self.model.load_weights(weights_filename[0])
        use_set = []
        for j, m_name in enumerate(self.m_names):
            if not np.any([ign in m_name for ign in self.ignore_tags]):
                if part_dict[m_name] == k_test:
                    use_set.append(j)
        inputs = np.take(self.all_data, use_set, axis=0)
        actuals = np.take(self.energy_list, use_set, axis=0)
        inputs = inputs.transpose([1, 0, 2])
        inputs = [atom for atom in inputs]
        sizes = np.take(self.sizes, use_set, axis=0)
        inputs.append(np.reciprocal(np.asarray(sizes).astype(float)))
        pred = self.model.predict(inputs)
        try:
            predictions.append(pred.flatten().tolist())
        except AttributeError:
            predictions.append(pred)

        predictions_average = np.mean(predictions, axis=0)
        diff = np.subtract(actuals, predictions_average)
        rmse = np.sqrt(np.mean(diff)**2)
        print('RMSE:', rmse)
        sstot = np.mean(actuals)
        ssres = np.sum(diff**2)
        r2 = 1 - np.divide(ssres, sstot)
        print('R-squared:', r2)

        plt.plot(actuals, actuals, color='k')
        plt.scatter(actuals, predictions_average,
                    s=5, color='r')
        plt.title('Predicted vs. Actual Energy')
        plt.ylabel('Predicted (meV)')
        plt.xlabel('Actual (meV)')
        plt.show()

    def plot_kfold_predictions(self, filename, weights_filenames):
        partitions_file = self.settings['partitions_file']
        libver = self.settings['libver']
        part_dict = read_partitions(partitions_file)
        k_list = sorted(set(part_dict.values()))

        data_dict = {}
        part = PartitionProcessor({})
        part.load_preprocessed(filename, libver=libver)

        part.load_partitions_from_file(partitions_file)

        for k in k_list:
            (system, train, test) = part.get_network_inputs(testing_tags=[k],
                                                            verbosity=0)
            data_dict[str(k)] = {}
            data_dict[str(k)]['inputs'] = test['inputs']
            data_dict[str(k)]['outputs'] = test['outputs']
            data_dict[str(k)]['sizes'] = test['sizes']
        preds = []
        actuals = []
        val_predictions = []
        for weights_file in weights_filenames:
            self.model.load_weights(weights_file)
            for k in k_list[1:]:
                inputs = data_dict[str(k)]['inputs']
                reference = data_dict[str(k)]['outputs']
                inputs = [atom for atom in inputs]
                sizes = data_dict[str(k)]['sizes']
                inputs.append(np.reciprocal(np.asarray(sizes).astype(float)))
                pred = self.model.predict(inputs)
                try:
                    preds.append(pred.flatten().tolist())
                except AttributeError:
                    preds.append(pred)
                actuals.append(reference)
            v_inputs = data_dict['-1']['inputs']
            v_inputs = [atom for atom in v_inputs]
            v_sizes = data_dict['-1']['sizes']
            v_inputs.append(np.reciprocal(np.asarray(v_sizes).astype(float)))
            val_predictions.append(self.model.predict(v_inputs).flatten())
        preds.append(np.mean(val_predictions, axis=0).tolist())
        actuals.append(data_dict['-1']['outputs'])

        act_comb = [j for i in actuals for j in i]
        pred_comb = [j for i in preds for j in i]
        diff = np.subtract(act_comb, pred_comb)
        rmse = np.sqrt(np.mean(diff**2))
        print('RMSE:', rmse)
        sstot = np.sum(np.subtract(act_comb, np.mean(act_comb))**2)
        ssres = np.sum(diff**2)
        r2 = 1 - np.divide(ssres, sstot)
        print('R-squared:', r2)
        plt.plot(act_comb, act_comb, color='k')
        cols = plt.cm.rainbow(np.linspace(0, 1, len(k_list)))
        labels = ['k={}'.format(k) for k in k_list]
        labels.append('validation')
        for act, pred, col, label in zip(actuals, preds, cols, labels):
            plt.scatter(act, pred, s=5, color=col, label=label)
        plt.title('Predicted vs. Actual Energy: \
        RMSE = {0:.4f} meV, R^2 = {1:.4f}'.format(rmse, r2))
        plt.legend()
        plt.ylabel('Predicted (meV)')
        plt.xlabel('Actual (meV)')
        plt.show()

    def visualize_model(self, filename):
        plot_model(self.model, to_file=filename,
                   show_shapes=True, rankdir='LR')


def plot_energy_predictions(pred_list, act_list, colors, legend, save=False,
                            filename='energy_predictions.png'):
    all_actual = [j for i in act_list for j in i]
    plt.plot(all_actual, all_actual, color='k')
    assert len(pred_list) == len(act_list)

    colors = colors[:len(act_list)]
    legend = legend[:len(act_list)]
    for act, pred, col, label in zip(act_list, pred_list, colors, legend):
        plt.scatter(act, pred, s=5, alpha=0.5, color=col, label=label)
    plt.legend()
    plt.title('Predicted vs. Actual Energy')
    plt.ylabel('Predicted (meV)')
    plt.xlabel('Actual (meV)')
    if save:
        plt.savefig(filename, dpi=600)
    else:
        plt.show()
