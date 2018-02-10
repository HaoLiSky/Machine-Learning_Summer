"""

Usage: >>>python train_v5.py

"""
import os
import sys
import h5py
import numpy as np
import keras.backend as K
import tensorflow as tf
from time import time
from keras.models import Model
from keras.layers import Input, Dense, Add, Dropout, Multiply
from keras.layers import ActivityRegularization
from keras.optimizers import Nadam, Adadelta, Adam, SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras.callbacks import TerminateOnNaN
# from keras.wrappers.scikit_learn import KerasClassifier
from keras.engine.topology import Layer
from sklearn.preprocessing import normalize
# from sklearn.model_selection import StratifiedKFold, cross_val_score
from itertools import combinations_with_replacement, product
from datetime import datetime
from configparser import ConfigParser

import matplotlib.pyplot as plt

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

reading_index = slice(None,None,None)

optimizer_dict = {'adam': Adam(),
                  'nadam': Nadam(),
                  'sgd': SGD(lr=0.0000001, decay=0,
                             momentum=0.6, nesterov=True),
                  'adadelta': Adadelta(clipnorm=1.)}

# seed for reproducibility
np.random.seed(1337)

pair_slice_choices = [[2],
                      [2, 6],
                      [0, 2, 4, 6],
                      slice(None, None, None)]

triplet_slice_choices = [[8],
                         [8, 9],
                         [8, 9, 26, 27],
                         [0, 1, 8, 9, 18, 19, 26, 27],
                         [0, 1, 4, 5, 8, 9, 12, 13, 18, 19, 22, 23, 26,
                          27, 30, 31],
                         slice(None, None, None)]

global stage, stage_time
stage = 0
stage_time = time()


def print_progress(line):
    global stage, stage_time
    if stage > 0:
        print('{0:.1f}s - {1}'.format(time() - stage_time,
                                      datetime.fromtimestamp(stage_time)
                                      .strftime('%I:%M:%S%p')))
    print('\n{}) {}'.format(stage, line))
    stage += 1
    stage_time = time()


def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


def spread(y_true, y_pred):
    return K.min(y_pred)-K.max(y_pred)


def mean_diff(y_true, y_pred):
    return K.mean(y_pred)-K.mean(y_true)

def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


class endmask(Layer):
    def __init__(self, **kwargs):
        super(endmask, self).__init__(**kwargs)

    def call(self, inputs):
        outputs = tf.where(tf.is_nan(inputs), K.zeros_like(inputs), inputs)
        return outputs


class LossHistory(Callback):
    def __init__(self):
        self.seen = 0
        self.display = 5  # display every x epochs
        self.test_losses = []
        self.train_losses = []

    # def on_train_begin(self, logs={}):
    # def on_batch_end(self, batch, logs={}):

    def on_epoch_end(self, epoch, logs={}):
        self.train_losses.append(logs.get('loss'))
        self.test_losses.append(logs.get('val_loss'))
        self.seen += 1
        if self.seen % self.display == 0:
            try:
                print('\r{}:'.format(self.seen).ljust(8),
                      '{0:.3f}'.format(logs.get('loss')).rjust(9),
                      '{0:.3f}'.format(logs.get('mean_diff')).rjust(9),
                      '{0:.3f}'.format(logs.get('spread')).rjust(9),
                      '   |',
                      '{0:.3f}'.format(logs.get('val_loss')).rjust(9),
                      '{0:.3f}'.format(logs.get('val_mean_diff')).rjust(9),
                      '{0:.3f}'.format(logs.get('val_spread')).rjust(9),

                      end='')
                sys.stdout.flush()
            except (TypeError, ValueError):
                pass


def normalize_inputs(unprocessed_data, Alpha, Beta):
    """

    Args:
        unprocessed_data: List of data sets (e.g. G1, G2)
            with equal-size first dimension.
        Alpha: number of interactions w.c. per data set
        Beta: number of parameter sets per data set

    Returns:
        processed_data: Flattened, normalized dataset.
            Length = per structure,
            width = feature vector (i.e. flattened and concatenated G1 and G2).

    """

    vectorized_fp_list = []
    for dataset, alpha, beta in zip(unprocessed_data, Alpha, Beta):
        fp_columned = [rearrange_to_columns(fp_per_s)
                       for fp_per_s in dataset]
        fp_lengths = [len(fingerprint) for fingerprint in
                      fp_columned]
        fp_as_grid = np.concatenate(fp_columned)

        normalized_grid = normalize(fp_as_grid, axis=0, norm='max')
        print('alpha={}, beta={}'.format(alpha, beta),
              '  min=', np.amin(normalized_grid),
              '  mean=', np.mean(normalized_grid),
              '  std=', np.std(normalized_grid),
              '  max=', np.amax(normalized_grid))

        regenerated_fps = regenerate_fp_by_len(normalized_grid, fp_lengths)
        # print(set([len(x) for x in regenerated_fps]))

        vectorized_fp_list.append(vectorize_fps(regenerated_fps, alpha))
    vectorized_fps = [np.concatenate((g1t, g2t), axis=1) for g1t, g2t
                      in zip(*vectorized_fp_list)]

    return vectorized_fps
    # for each fingerprint,
    # length = # structures * # combinations w.r. of interactions
    # shape = ((S * j), k)

    # regenerate data per structure by slicing based on recorded atoms
    # per structure and then flatten for one 1d vector per structure


def rearrange_to_columns(fingerprint):
    layers = [layer for layer in np.asarray(fingerprint)]
    joined = np.concatenate(layers, axis=0)
    return joined


def regenerate_fp_by_len(fp_as_grid, fp_lengths):
    indices = np.cumsum(fp_lengths)[:-1]
    return np.split(fp_as_grid, indices)


def vectorize_fps(regenerated_fps, alpha):
    lengths = [len(fingerprint) for fingerprint in regenerated_fps]
    vectors = [[atom.flatten(order='F')
                for atom in np.split(grid,
                                     np.arange(alpha, length, alpha))]
               for grid, length in zip(regenerated_fps,
                                       lengths)]
    return vectors


def pad_fp_by_element(input_data, compositions, final_layers):
    """

    Slice fingerprint arrays by elemental composition and
    arrange slices onto empty* arrays to produce sets of padded fingerprints
    (i.e. equal length for each type of fingerprint)

    *padding is set to a constant -1, to be used in Keras' masking function.

    Args:
        input_data: 2D array of structures/features.
            i.e. output of normalize_inputs()
        compositions: list of "layer" heights (per element) in # of atoms.
        final_layers: list of desired "layer" heights (per element)
            in # of atoms.
    Returns:
        output_data: 2D array of structures/features, padded to equal lengths
            in the 2nd dimension (equal number of atoms per structure).

    """
    new_input_data = []
    for structure, initial_layers in zip(input_data, compositions):
        assert len(final_layers) == len(initial_layers)
        data = np.asarray(structure)
        data_shape = data.shape
        natoms_i = data_shape[0]
        natoms_f = sum(final_layers)
        natoms_diff = natoms_f - natoms_i
        secondary_dims = len(data_shape) - 1
        pad_widths = [(natoms_diff, 0)] + [(0, 0)] * secondary_dims
        # tuple of (header_pad, footer_pad) per dimension
        # if masking:
        data_f = np.pad(np.ones(data.shape), pad_widths, 'edge') * 0.0
        # else:
        # data_f = np.pad(np.zeros(data.shape), pad_widths, 'edge')
        slice_pos = np.cumsum(initial_layers)[:-1]
        # row indices to slice to create n sections
        data_sliced = np.split(data, slice_pos)

        start_pos = np.insert(np.cumsum(final_layers)[:-1], 0, 0)
        # row indices to place sections of correct length

        for sect, start in zip(data_sliced, start_pos):
            end = start + len(sect)
            data_f[start:end, ...] = sect
        new_input_data.append(np.asarray(data_f))
    return np.asarray(new_input_data)


def element_model(input_vector_length, n_hlayers, dense_units,
                  element, activation='softplus', dropout=0, l1=0, l2=0):
    """

    Args:
        input_vector_length: Length of feature vector.
        n_hlayers: Number of hidden layers.
        dense_units: Number of neurons per hidden layer.
        name: Element label.
        activation: Activation function.

    Returns:
        model: Keras model of element, to be called in structure_model().

    """

    input_layer = Input(shape=(input_vector_length,),
                        name='{}_inputs'.format(element))
    # first layer (input)

    h_layer = input_layer
    for i in range(n_hlayers):
        # stack hidden layers
        h_layer = Dense(dense_units, activation=activation,
                        name='{}_hidden_{}'.format(element, i))(h_layer)
        if l1 > 0 or l2 > 0:
            h_layer = ActivityRegularization(
                l1=l1, l2=l2, name='{}_reg_{}'.format(element, i))(h_layer)
        if dropout > 0:
            h_layer = Dropout(
                dropout, name='{}_dropout_{}'.format(element, i))(h_layer)

    output_layer = Dense(1,
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

    # model = Model(inputs=structure_input_lanes,
    #               outputs=total_energy_output)

    n_atoms_input = Input(shape=(1,), name='1/atoms')
    structure_input_lanes.append(n_atoms_input)
    avg_output = Multiply()([n_atoms_input, total_energy_output])
    # masked_output = endmask(name='{}_mask'.format(name))(avg_output)
    # # Sum up atomic energies

    model = Model(inputs=structure_input_lanes,
                  outputs=avg_output)

    return model


def read_fingerprints(filename, index):
    with h5py.File(filename, 'r',
                   libver='latest') as h5f:
        s_names = [line.split(b';')[0].decode('utf-8')
                   for line in h5f['system']['sys_entries'][()]][index]

        if os.path.isfile('badlog'):
            with open('badlog', 'r') as fil:
                badlog = fil.read().split('\n')
                badlog = [x.replace('.poscar', '') for x in badlog]
            print(len(s_names), end=' to ')
            s_names = [x for x in s_names
                       if x.split('_')[0] not in badlog]
            print(len(s_names))
        # read list of names from system/sys_entries dataset
        # slice by index
        g_1 = []
        g_2 = []
        energy_list = []
        sys_elements = [symbol.decode('utf-8')
                        for symbol in h5f['system'].attrs['sys_elements']]
        # get master set of elements from system attributes
        element_counts_list = []
        s_dset = h5f['structures']  # top-level group reference
        for j, s_name in enumerate(s_names):
            # loop over structures
            dset = s_dset[s_name]  # group reference for one structure
            n_atoms = dset.attrs['natoms']
            element_set = [symbol.decode('utf-8')
                           for symbol in dset.attrs['element_set']]
            # set of elements per structure
            if not set(element_set).issubset(set(sys_elements)):
                continue  # skip if not part of master set

            count_per_element = dset.attrs['element_counts']
            energy_value = float(dset.attrs['energy']) / n_atoms * 1000
            '''
            atomic_energy= np.divide(energy_value,
                                     sum(count_per_element))
            if atomic_energy < lower_bound or atomic_energy > upper_bound:
                print('skipped {}; E/n = {}'.format(s_name, atomic_energy))
                continue
            '''
            g_1.append(dset['G_1'][()])
            g_2.append(dset['G_2'][()])
            energy_list.append(energy_value)
            element_counts_list.append(count_per_element.tolist())
        print('Read {} fingerprints from {}'.format(len(energy_list),
                                                    filename))

    return g_1, g_2, energy_list, element_counts_list, sys_elements


def grid_search_train(index, loss, options, hyperparameters):
    filename = options['fingerprints_file']

    optimizer = options['optimizer']
    epochs = options['epochs']
    batch_size = options['batch_size']
    split_positions = options['split_positions']
    restart = options['restart']

    if 'padded' not in filename:
        print_progress('reading data')
        read_data = read_fingerprints(filename, index)
        do_preprocess = True
    else:
        do_preprocess = False

    hyperparameter_sets = [hyperparameters['H'],
                           hyperparameters['D'],
                           hyperparameters['activations'],
                           hyperparameters['a'],
                           hyperparameters['b'],
                           hyperparameters['dropouts'],
                           hyperparameters['l1_regs'],
                           hyperparameters['l2_regs'],
                           hyperparameters['k_test_ind']]

    for (n_hlayers, dense_units,
         activation, n_pair_params,
         n_triplet_params, dropout,
         l1_reg, l2_reg, k_test_ind) in list(product(*hyperparameter_sets)):
        global stage
        stage = 1

        tag = '{},{}-{},{}-{},{}-{}-{},{}'.format(activation, n_hlayers,
                                                  dense_units,
                                                  n_pair_params,
                                                  n_triplet_params,
                                                  dropout, l1_reg, l2_reg,
                                                  k_test_ind)

        if do_preprocess:
            print_progress('preprocessing fingerprints: ' + tag)
            preprocess(filename.replace('.hdf5', '_padded.hdf5'),
                       read_data, n_pair_params, n_triplet_params)
            partitioned_data = read_preprocessed(
                filename.replace('.hdf5', '_padded.hdf5'),
                k_test_ind, split_positions, n_pair_params, n_triplet_params)
        else:
            partitioned_data = read_preprocessed(filename,
                                                 k_test_ind, split_positions,
                                                 n_pair_params,
                                                 n_triplet_params)
        print_progress('training model: ' + tag)
        train(partitioned_data,
              n_hlayers, dense_units, activation,
              epochs, batch_size, loss, optimizer,
              dropout, l1_reg, l2_reg, tag, restart)


def read_preprocessed(filename, k_test_ind, split_positions,
                      n_pair_params, n_trip_params):
    with h5py.File(filename, 'r', libver='latest') as h5f:
        sys = h5f['system']
        energy_list = sys.attrs['s_energies']
        element_counts_list = sys.attrs['s_element_counts']
        sys_elements = [x.decode() for x in sys.attrs['sys_elements']]
        padded = h5f['{}-{}'.format(n_pair_params, n_trip_params)][()]

    n_list = np.sum(element_counts_list, axis=1)

    max_per_element = np.amax(element_counts_list,
                              axis=0)  # maximum occurrences per atom type

    assert not np.any(np.isnan(padded))
    assert not np.any(np.isinf(padded))

    chunks = np.split(padded, split_positions)
    valid_inputs = chunks.pop(k_test_ind).transpose([1, 0, 2])
    train_inputs = np.concatenate(chunks).transpose([1, 0, 2])
    print(train_inputs.shape[1], valid_inputs.shape[1])

    chunked_names = np.split(n_list, split_positions)
    n_list_valid = chunked_names.pop(k_test_ind)
    n_list_train = np.concatenate(chunked_names)

    print(len(n_list_train), len(n_list_valid))

    chunked_energy = np.split(energy_list, split_positions)
    valid_outputs = chunked_energy.pop(k_test_ind)
    train_outputs = np.concatenate(chunked_energy)

    print(len(train_outputs), len(valid_outputs))

    chunked_compositions = np.split(element_counts_list, split_positions)
    valid_compositions = chunked_compositions.pop(k_test_ind)
    valid_compositions = [','.join([str(x) for x in y])
                                    for y in valid_compositions]
    print(set(valid_compositions))

    return (train_inputs, valid_inputs,
            train_outputs, valid_outputs,
            sys_elements, max_per_element, n_list_train, n_list_valid)


def preprocess(filename, read_data,
               n_pair_params, n_trip_params):
    g_1, g_2, energy_list, element_counts_list, sys_elements = read_data

    pair_slice = pair_slice_choices[int(np.log2(n_pair_params))]
    triplet_slice = triplet_slice_choices[int(np.log2(n_trip_params))]

    g_1 = [g[..., pair_slice] for g in g_1]
    g_2 = [g[..., triplet_slice] for g in g_2]
    fp_read = [g_1, g_2]  # rearrange by fingerprint

    max_per_element = np.amax(element_counts_list,
                              axis=0)  # maximum occurrences per atom type

    pair_interactions = list(combinations_with_replacement(sys_elements, 1))
    triplet_interactions = list(combinations_with_replacement(sys_elements, 2))
    a = [len(pair_interactions), len(triplet_interactions)]  # per fingerprint
    b = [n_pair_params, n_trip_params]  # per fingerprint
    normalized = normalize_inputs(fp_read, a, b)

    # print('prepadded shapes:', set([x.shape for x in normalized]))

    padded = pad_fp_by_element(normalized,
                               element_counts_list,
                               max_per_element)

    with h5py.File(filename, 'w', libver='latest') as h5f:
        sys = h5f.require_group('system')
        sys.attrs['s_energies'] = energy_list
        sys.attrs['s_element_counts'] = element_counts_list
        sys.attrs['sys_elements'] = np.string_(sys_elements)
        h5f.create_dataset('{}-{}'.format(n_pair_params, n_trip_params),
                           data=padded, shape=padded.shape)

    print(padded.shape)


def train(data, n_hlayers, dense_units, activation,
          epochs, batch_size, loss, optimizer,
          dropout, l1_reg, l2_reg, filename, restart):
    (train_inputs, valid_inputs,
     train_outputs, valid_outputs,
     sys_elements, max_per_element, n_list_train, n_list_valid) = data

    global layers_id
    layers_id = 0
    input_vector_length = train_inputs.shape[-1]

    element_models = [element_model(input_vector_length, n_hlayers,
                                    dense_units, specie,
                                    activation=activation,
                                    dropout=dropout,
                                    l1=l1_reg,
                                    l2=l2_reg)
                      for specie in sys_elements]

    model = structure_model(input_vector_length, max_per_element,
                            sys_elements, element_models)
    model.compile(loss=loss, optimizer=optimizer, metrics=[mean_pred, spread])
    model.summary()

    train_inputs = [atom for atom in train_inputs]
    valid_inputs = [atom for atom in valid_inputs]

    valid_inputs.append(np.reciprocal(np.asarray(n_list_valid)
                                      .astype(float)))
    train_inputs.append(np.reciprocal(np.asarray(n_list_train)
                                      .astype(float)))

    print('Training structure sizes:', set(n_list_train.tolist()))
    print('Testing structure sizes:', set(n_list_valid.tolist()))

    a = model.predict(train_inputs)
    assert not np.any(np.isnan(a))
    assert not np.any(np.isinf(a))

    checkpointer = ModelCheckpoint(filepath='c_' + filename + '.h5',
                                   verbose=0,
                                   monitor='val_loss',
                                   save_weights_only=False,
                                   period=50, save_best_only=True)

    #es = EarlyStopping(min_delta=0.01, patience=5000)

    history = LossHistory()
    nan_check = TerminateOnNaN()

    if os.path.isfile('c_{}.h5'.format(filename)) and restart:
        print('loaded weights')
        model.load_weights('c_{}.h5'.format(filename))

    print('\n\n', ' ' * 9,
          'Training'.center(29), '|  ', 'Testing'.center(29))
    print('Epoch'.ljust(7),
          'rmse'.rjust(9), 'mean pred'.rjust(9), 'spread'.rjust(9),
          '   |',
          'rmse'.rjust(9), 'mean pred'.rjust(9), 'spread'.rjust(9))

    try:
        model.fit(train_inputs, train_outputs,
                  epochs=epochs, verbose=1,
                  validation_data=(valid_inputs, valid_outputs),
                  batch_size=batch_size,
                  callbacks=[nan_check, checkpointer, #es
                             history])
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        pass
    model.save_weights('final_{}.h5'.format(filename))

    # print(model.get_weights())

    train_hist = history.train_losses
    test_hist = history.test_losses

    print(len(train_hist), len(test_hist))

    fileunique = (filename
                  + (datetime.fromtimestamp(time()).strftime('%H%M%S')))

    np.savetxt(fileunique + '.csv',
               np.vstack((train_hist, test_hist)).transpose(),
               delimiter=',', newline=';\n')

    # f = plt.figure()
    # ax1 = f.add_subplot(211, aspect='equal')
    '''

               
    e_pred = model.predict([atom for atom in valid_inputs]).flatten()

    e_act = np.asarray([atom for atom in valid_outputs])

    rmse = np.mean(np.sqrt(np.divide(np.subtract(e_act,
                                                 e_pred),
                                     n_list_valid) ** 2))
                                     
    f, (ax1, ax2) = plt.subplots(1, 2)
    ax1.set_title('{}, RMSE = {}'.format(filename, rmse))
    ax1.plot(e_act, e_act, color='k')
    ax1.scatter(e_act, e_pred, color='g')
    ax1.set_xlabel('Actual energy [eV/atom]')
    ax1.set_ylabel('Predicted energy [eV/atom]')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('RMSE Loss [eV/atom]')
    
    # plot_model(model, to_file=fileunique + '.png')
    # ax2 = f.add_subplot(212, aspect='equal')
    plt.figure()
    ax2 = plt.subplot(111)
    ax2.set_title('Testing/Training Losses' + str(np.mean(test_hist[-100:])))
    ax2.plot(train_hist, color='r', label='Training')
    ax2.plot(test_hist, color='b', label='Testing')

    plt.yscale('symlog')
    plt.grid(True)
    plt.gca().yaxis.grid(True, which='minor')
    plt.legend()

    # f.set_tight_layout(True)

    plt.savefig(fileunique + '.png', dpi=600)

    plt.clf()
    '''

def get_settings(filename):
    parser = ConfigParser()
    parser.read(filename)
    gs = parser['Grid-Search']
    options = {}
    hyperparameters = {}

    options['fingerprints_file'] = gs.get('fingerprints_file')

    options['batch_size'] = int(gs.get('batch_size', fallback='1024'))
    options['epochs'] = int(gs.get('epochs', fallback='5000'))
    options['optimizer'] = optimizer_dict[gs.get('optimizer',
                                                 fallback='nadam')]
    options['split_positions'] = [int(x)
                                  for x in
                                  gs.get('split_positions').split(';')]

    hyperparameters['k_test_ind'] = [int(x)
                                     for x in gs.get('k_test_ind',
                                                     fallback='1').split(';')]

    hyperparameters['activations'] = gs.get('activations',
                                            fallback='softplus').split(';')
    hyperparameters['dropouts'] = [float(x)
                                   for x in gs.get('dropouts',
                                                   fallback='0').split(';')]
    hyperparameters['l1_regs'] = [float(x)
                                  for x in gs.get('l1_regs',
                                                  fallback='0').split(';')]
    hyperparameters['l2_regs'] = [float(x)
                                  for x in gs.get('l2_regs',
                                                  fallback='0').split(';')]
    hyperparameters['H'] = [int(x) for x in gs.get('H',
                                                   fallback='1').split(';')]
    hyperparameters['D'] = [int(x) for x in gs.get('D',
                                                   fallback='1').split(';')]
    hyperparameters['a'] = [int(x) for x in gs.get('a',
                                                   fallback='1').split(';')]
    hyperparameters['b'] = [int(x) for x in gs.get('b',
                                                   fallback='1').split(';')]

    print(options['fingerprints_file'], options['batch_size'],
          options['epochs'])
    print(options['optimizer'], options['split_positions'],
          hyperparameters['k_test_ind'])
    print(hyperparameters['dropouts'], hyperparameters['l1_regs'],
          hyperparameters['l2_regs'], hyperparameters['activations'])
    print(hyperparameters['H'], hyperparameters['D'], hyperparameters['a'],
          hyperparameters['b'])

    return options, hyperparameters


if __name__ == '__main__':
    try:
        filename = sys.argv[1]
        assert os.path.isfile(filename)
    except (IOError, AssertionError):
        raise IOError('failed to open options file.')

    options, hyperparameters = get_settings(filename)

    try:
        options['restart'] = sys.argv[2] == 'restart'
    except (IndexError, KeyError):
        options['restart'] = False

    grid_search_train(reading_index, rmse_loss, options, hyperparameters)
