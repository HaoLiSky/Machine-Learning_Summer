"""
Network for training/testing using Keras.
"""
import os
import argparse
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning)
import numpy as np
import math
from time import time
from datetime import datetime
from itertools import product
from keras.models import Model
from keras import initializers
from keras.layers import Input, Dense, Add, Dropout, Multiply
from keras.layers import ActivityRegularization
from keras.optimizers import Nadam, Adadelta, Adam, SGD
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras.callbacks import TerminateOnNaN
from nnf.io_utils import generate_tag, SettingsParser
from nnf.custom_keras_utils import spread, mean_pred, LossTracking, rmse_loss, custom_sigmoid
from nnf.batch_preprocess import PartitionProcessor


activation_list = ['softplus',
                   'relu',
                   'tanh',
                   'sigmoid',
                   'softplus']
optimizer_list = [SGD(lr=0.001, decay=0.001,
                      momentum=0.9, nesterov=True),
                  Adam(),
                  Nadam(),
                  Adadelta(),
                  Adam(clipnorm=1.0),
                  Nadam(clipnorm=1.0),
                  Adadelta(clipnorm=1.0),
                  'rmsprop',
                  'adagrad',
                  'adamax']
loss_list = [rmse_loss,
             'mean_squared_error',
             'mean_absolute_error',
             'mean_absolute_percentage_error']
# arbitrary fixed seed for reproducibility
np.random.seed(8)
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'


def print_progress(line):
    """
    Simple printing utility with timestamps.
    """
    global stage, stage_time
    if stage > 0:
        print('{0:.1f}s - {1}'.format(time() - stage_time,
                                      datetime.fromtimestamp(stage_time)
                                      .strftime('%I:%M:%S%p')))
    print('\n{}) {}'.format(stage, line))
    stage += 1
    stage_time = time()


def element_model(element, input_vector_length, n_hlayers, dense_units,
                  activation, dropout, l1, l2):
    """
    Args:
        element: Element label.
        input_vector_length: Length of feature vector.
        n_hlayers: Number of hidden layers.
        dense_units: Number of neurons per hidden layer.
        activation: Activation function.
        dropout: Dropout fraction.
        l1: L1 weight regularization penalty(i.e. LASSO regression).
        l2: L2 weight regularization penalty (i.e. ridge regression).

    Returns:
        model: Keras model of element, to be called in structure_model().
    """

    input_layer = Input(shape=(input_vector_length,),
                        name='{}_inputs'.format(element))
    # first layer (input)

    h_layer = input_layer
    for i in range(n_hlayers):
        # stack hidden layers
        h_layer = Dense(dense_units, activation=custom_sigmoid,
                        kernel_initializer = initializers.TruncatedNormal(mean=0.0, stddev=0.05, seed=None),
                        use_bias = False,
                        name='{}_hidden_{}'.format(element, i))(h_layer)
        if l1 > 0 or l2 > 0:
            h_layer = ActivityRegularization(
                    l1=l1, l2=l2, name='{}_reg_{}'.format(element, i))(h_layer)
        if dropout > 0:
            h_layer = Dropout(
                    dropout, name='{}_dropout_{}'.format(element, i))(h_layer)

    output_layer = Dense(1,
                         use_bias = False,
                         name='{}_Ei'.format(element))(h_layer)

    model = Model(input_layer, output_layer,
                  name='{}_network'.format(element))

    return model


def structure_model(input_vector_length, count_per_element, sys_elements,
                    element_models):
    """
    Args:
        count_per_element: Maximum occurences of each element
            i.e. number of input neurons per element.
        sys_elements: List of element symbols as strings.
        element_models: List of element models
            i.e. output of element_model().
        input_vector_length (int): Length of flattened input vector.

    Returns:
        Keras model to compile and train with padded data.
    """
    global layers_id  # global count for unique naming per layer
    assert len(count_per_element) == len(element_models)
    structure_input_lanes = []
    structure_output_lanes = []
    for n_atoms, specie_model, name in zip(count_per_element, element_models,
                                           sys_elements):
        # loop over element types present
        specie_input_lanes = [Input(shape=(input_vector_length,),
                                    name='{}_Input-{}'.format(name, i))
                              for i in range(n_atoms)]
        # one Input vector expected per atom

        specie_output_lanes = [specie_model(input_lane)
                               for input_lane in specie_input_lanes]
        # call corresponding element model for each atom
        structure_input_lanes.extend(specie_input_lanes)
        structure_output_lanes.extend(specie_output_lanes)

    total_energy_output = Add(name='E_tot')(structure_output_lanes)

    #model = Model(inputs=structure_input_lanes,
    #              outputs=total_energy_output)

    n_atoms_input = Input(shape=(1,), name='1/atoms')
    structure_input_lanes.append(n_atoms_input)
    avg_output = Multiply()([n_atoms_input, total_energy_output])         #set the output as the energy per atom (not the total energy!)#
    # masked_output = endmask(name='{}_mask'.format(name))(avg_output)
    # # Sum up atomic energies

    model = Model(inputs=structure_input_lanes,
                  outputs=avg_output)

    return model


class Network:
    """
    Artificial Neural Network.
    """
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)
        self.checkpoint_name = 'c{}.h5'
        self.loss_history_name = 'l{}.csv'
        self.final_weights_name = 'w{}.h5'
        self.verbosity = self.settings['verbosity']
        self.record = self.settings['record']

    def load_data(self, filename, part_file, run_settings):
        """
        Args:
            filename (str): File with preprocessed fingerprint data.
                e.g. fingerprints.h5
            part_file (str): File with partition designations for each
                fingerprint identifier.
            run_settings (dict): Run-specific settings, i.e. including
                a unique value of k_test for k-fold cross validation.
        """
        libver = self.settings['libver']
        test_tags = run_settings['test_chunk']
        part = PartitionProcessor({})
        part.load_preprocessed(filename, libver=libver)
        part.load_partitions_from_file(part_file)
        (system,
         self.train,
         self.test) = part.get_network_inputs(testing_tags=test_tags)

        self.sys_elements = [val.decode('utf-8')
                             for val in system['sys_elements']]
        self.max_per_element = system['max_per_element']

    def train_network(self, run_settings, tag=None):
        """
        Args:
            run_settings (dict): SettingsParser for training. Can use
                Network.settings or can be specified, as in grid_search.
            tag (str): Optional run identifier (e.g. 'ridge_test_2'), used
                in output filenames. If none is given, the values in
                run_settings are encoded into an alphanumeric string.
        """
        self.layers_id = 0
        input_vector_length = self.train['inputs'].shape[-1]
        epochs = run_settings['epochs']
        batch_size = run_settings['batch_size']
        n_hlayers = run_settings['hiddenlayer']
        dense_units = run_settings['hiddenneuron']
        activation = activation_list[run_settings['activation']]
        dropout = run_settings['dropout']
        l1 = run_settings['l1']
        l2 = run_settings['l2']
        optimizer = optimizer_list[run_settings['optimizer']]
        loss = loss_list[run_settings['loss']]
        save_best = self.settings['checkpoint_best_only']
        save_period = self.settings['checkpoint_period']
        allow_restart = self.settings['allow_restart']
        # 0) generate identifier string from settings if none specified
        if not tag or tag == 'None':
            tag = generate_tag(run_settings)
            j = 0
            while os.path.isfile(self.checkpoint_name.format(tag)):
                j += 1
                tag = generate_tag(run_settings, add_on=str(j))
            print('Run identifier:', tag)
        # 1) define elemental subnetworks
        element_models = [element_model(specie,
                                        input_vector_length,
                                        n_hlayers,
                                        dense_units,
                                        activation,
                                        dropout, l1, l2)
                          for specie in self.sys_elements]
        # 2) create molecule-level network
        model = structure_model(input_vector_length, self.max_per_element,
                                self.sys_elements, element_models)
        model.compile(loss=loss, optimizer=optimizer,
                      metrics=[mean_pred, spread])
        if self.verbosity >= 2:
            model.summary()
        # 3) merge molecule fingerprint and size data into columns
        training_inputs = [atom for atom in self.train['inputs']]
        testing_inputs = [atom for atom in self.test['inputs']]
        training_inputs.append(np.reciprocal(np.asarray(self.train['sizes'])
                                             .astype(float)))
        testing_inputs.append(np.reciprocal(np.asarray(self.test['sizes'])
                                            .astype(float)))
        # 4) ensure model does not return infinity or NaN values
        # if so, there is, most likely, a problem with the input data
        #untrained_sample = model.predict(training_inputs)
        #assert not np.any(np.isnan(untrained_sample))
        #assert not np.any(np.isinf(untrained_sample))
        # 5) define Keras callbacks that run alongside training
        history = LossTracking()
        nan_check = TerminateOnNaN()
        callbacks = [history, nan_check]
        if self.record > 0:
            checkpoint_name = self.checkpoint_name.format(tag)
            checkpointer = ModelCheckpoint(filepath=checkpoint_name,
                                           verbose=0,
                                           monitor='val_loss',
                                           save_weights_only=False,
                                           period=save_period,
                                           save_best_only=save_best)
            callbacks.append(checkpointer)
        # 6) load weights from previous runs if allow
        if os.path.isfile(self.checkpoint_name.format(tag)) and allow_restart:
            print('Loaded previous run.')
            model.load_weights(self.checkpoint_name.format(tag))
        # 7) train model
        print('\n', ' ' * 9,
              'Training'.center(29), '|  ', 'Testing'.center(29))
        print('Epoch'.ljust(7),
              'rmse/meV'.rjust(9), 'mean-pred/meV'.rjust(9), 'spread/meV'.rjust(9),
              '   |',
              'rmse/meV'.rjust(9), 'mean-pred/meV'.rjust(9), 'spread/meV'.rjust(9), '\n')
        try:
            model.fit(training_inputs, self.train['outputs'],
                      epochs=epochs, verbose=0,
                      validation_data=(testing_inputs,
                                       self.test['outputs']),
                      batch_size=batch_size,
                      callbacks=callbacks)
            #summary = model.summary()
            #print(summary)

            weights = model.get_weights()
            print(weights)

            first_weights = weights[0]    #reminder: this is 2D array#
            #input_number = len(weights[0])
            #hidden_neuron_number = len(weights[0][0])

            second_weights = weights[1]   #reminder: this is 1D array#
            #first_bias = weights[1]       #reminder: this is 2D array#
            #second_bias = weights[3]      #reminder: this is 1D array#
            
            np.savetxt('first_weights.csv', first_weights, delimiter = ',')  
            np.savetxt('second_weights.csv', second_weights, delimiter = ',')
            #np.savetxt('first_bias.csv', first_bias, delimiter = ',')
            #np.savetxt('second_bias.csv', second_bias, delimiter = ',')

            #G_weights_sum = []
 
            #for i in range(0, hidden_neuron_number):
            
             #         G_weights_sum.append(np.sum(np.multiply(G, first_weights[:,i])))
                        
            #fianl_output = []


            #for i in range(0, hidden_neuron_number):

             #         final_output[i] = second_weights[i] * (1 / (1+ math.exp(G_weights_sum[i] + first_bias[i])))
         
            #sum_of_all_terms = np.sum(final_output)

            #energy_per_atom = sum_of_all_terms + second_bias

            #print(first_weights)
            #print(second_weights)
            #print(first_bias)
            #print(second_bias)
            #print('number of input: ', input_number)
            #print('number of hidden neuron: ', hidden_neuron_number) 



            #print(weighttest)
            #print(realbias)

        except (KeyboardInterrupt, SystemExit):
            print('\n')
        # 8) record weights and history of losses
        train_hist = history.train_losses
        test_hist = history.test_losses
        if self.record > 0:
            model.save_weights(self.final_weights_name.format(tag))
            np.savetxt(self.loss_history_name.format(tag),
                       np.vstack((train_hist, test_hist)).transpose(),
                       delimiter=',', newline=';\n')
        return test_hist[-1]

    def grid_search(self, settings_file, filename):
        """
        Grid search for parameters/hyperparameters using network.

        Args:
            settings_file (str): File with GridSearch setting ranges.
            filename (str): Output .csv file with entry data and resulting
                testing losses.
        """
        settings_set = SettingsParser('GridSearch').read(settings_file)
        keys = list(settings_set.keys())
        values = list(settings_set.values())
        range_keys = [keys[j]
                      for j, range_ in enumerate(values)
                      if len(range_) > 1]
        print('\nGrid Search:', ','.join(range_keys))

        setting_combinations = list(product(*values))

        run_entries = []
        for i, setting_combination in enumerate(setting_combinations):
            print('\n\nRun {}:'.format(i+1), end='   ')
            run_settings = {k: v for k, v in zip(keys, setting_combination)}
            entry = [str(val) for val in setting_combinations]
            for key in range_keys:
                print('{} = {}'.format(key, run_settings[key]), end='   ')
            self.load_data(run_settings['inputs_name'],
                           run_settings['partitions_file'],
                           run_settings)
            run_testing_loss = self.train_network(run_settings,
                                                  tag='_'+str(i+1))
            entry.append('{0:.4f}'.format(run_testing_loss))
            run_entries.append(entry)

        with open(filename, 'w') as search_record:
            lines = [','.join(entry) + ';\n' for entry in run_entries]
            lines.insert(0, ','.join(keys) + ';\n')
            search_record.writelines(lines)


if __name__ == '__main__':
    description = 'Train an artificial neural network.'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('--settings_file', '-s', default='settings.cfg',
                           help='Filename of settings.')
    argparser.add_argument('--verbosity', '-v', default=0, action='count')
    argparser.add_argument('--grid', '-g', action='store_true',
                           help='Begin grid search.')

    args = argparser.parse_args()
    settings = SettingsParser('Network').read(args.settings_file)
    settings['verbosity'] = args.verbosity

    input_name = settings['inputs_name']
    partitions_file = settings['partitions_file']
    output_name = settings['outputs_name']

    network = Network(settings)
    if args.grid:
        network.grid_search(args.settings_file, output_name)
    else:
        network.load_data(input_name, partitions_file, settings)
        final_loss = network.train_network(settings)
        print('Final loss:', final_loss)
        print(model.save_weights(self.final_weights_name.format(tag)))       


