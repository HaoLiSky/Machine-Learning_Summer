"""

Usage: >>>python train_v3.py gold_simple_fps.hdf5

"""
# If it displays "nan" loss (which is a common problem in Tensorflow), solve it by changing the float value in ~/.keras/keras.json from "float32" to "float64" #

import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import h5py
import numpy as np
import keras.backend as K
from time import time
from keras.models import Model
from keras.layers import Input, Dense, Add
from keras.optimizers import SGD, Adam
from keras.callbacks import ModelCheckpoint, EarlyStopping, Callback
# from keras.callbacks import TerminateOnNaN
from sklearn.preprocessing import normalize
from itertools import combinations_with_replacement, product

#################################################
#   ___  __    ___    ___         __
#  |__  |  \ |  |      |  |__| | /__`        |
#  |___ |__/ |  |      |  |  | | .__/        |
#                                            v

reading_index = slice(None, None, None)
# starting index, ending index, step for reading data from .hdf5
# e.g. (1, 100, 2) for the first 50 odd indices
# e.g. (None, None, None) to use entire dataset

set_pair_params = [8]  # range: [1, 8]
set_trip_params = [32]  # range: [1, 32]

set_tt_split = [0.8]  # fraction
set_hlayers = [2]  # number of hidden layers
set_dense_units = [40]  # number of neurons per hidden layer
set_activation = [None, 'linear', 'sigmoid', 'softplus']

epochs = 5000
batch_size = 256
#optimizer = Adam()
optimizer = SGD(lr=0.0000001, decay=1e-6, momentum=0.9, nesterov=True)

#   ___  __    ___    ___         __         ^
#  |__  |  \ |  |      |  |__| | /__`        |
#  |___ |__/ |  |      |  |  | | .__/        |
#
#################################################

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
        print('{0:.1f} seconds elapsed.'.format(time() - stage_time))
    print('\n{}) {}'.format(stage, line))
    stage += 1
    stage_time = time()


def mean_pred(y_true, y_pred):
    return K.mean(y_pred)


def rmse_loss(y_true, y_pred):
    return K.sqrt(K.mean(K.square(y_pred - y_true), axis=-1))


class LossHistory(Callback):
    def __init__(self, display=10):
        self.seen = 0
        self.display = display

    def on_train_begin(self, logs={}):
        self.losses = []

    def on_batch_end(self, batch, logs={}):
        self.losses.append(logs.get('loss'))

    def on_epoch_end(self, epoch, logs={}):
        self.seen += 1
        if self.seen % self.display == 0:
            print('\rEpoch {0} Loss: {1:.1f} meV'.ljust(70)
                  .format(self.seen, logs.get('loss') * 1000),
                  end='')


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
        #     data_f = np.pad(np.ones(data.shape), pad_widths, 'edge') * -1
        # else:
        data_f = np.pad(np.zeros(data.shape), pad_widths, 'edge')
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
                  name, activation='softplus'):
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
    global layers_id  # global count for unique naming per layer

    input_layer = Input(shape=(input_vector_length,),
                        name=name + '_' + str(
                            layers_id))  # first layer (input)
    layers_id = layers_id + 1

    h_layer = input_layer
    for i in range(n_hlayers):
        # stack hidden layers
        h_layer = Dense(dense_units, activation=activation)(h_layer)
    output_layer = Dense(1)(h_layer)

    model = Model(input_layer, output_layer, name=name)
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
                                    name=name + '_' + str(layers_id + i))
                              for i in range(n_atoms)]
        # one Input vector expected per atom
        layers_id += n_atoms
        specie_output_lanes = [specie_model(input_lane)
                               for input_lane in specie_input_lanes]
        # call corresponding element model for each atom
        try:
            structure_input_lanes += [j for i in specie_input_lanes for j in i]
        except TypeError:  # raised in unary system
            structure_input_lanes += specie_input_lanes
        structure_output_lanes += specie_output_lanes
    total_energy_output = Add()(structure_output_lanes)
    # Sum up atomic energies

    model = Model(inputs=structure_input_lanes, outputs=total_energy_output)
    return model


def read_fingerprints(filename, index):
    with h5py.File(filename, 'r',
                   libver='latest') as h5f:
        s_names = [line.split(b';')[0].decode('utf-8')
                   for line in h5f['system']['sys_entries'][()]][index]
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
            # number_of_atoms = dset.attrs['natoms']
            element_set = [symbol.decode('utf-8')
                           for symbol in dset.attrs['element_set']]
            # set of elements per structure
            if not set(element_set).issubset(set(sys_elements)):
                continue  # skip if not part of master set

            count_per_element = dset.attrs['element_counts']
            energy_value = dset.attrs['energy']
            g_1.append(dset['G_1'][()])
            g_2.append(dset['G_2'][()])
            energy_list.append(float(energy_value))
            element_counts_list.append(count_per_element.tolist())
            # print('read', j, end='\r')
    return g_1, g_2, energy_list, element_counts_list, sys_elements


def grid_search_train(filename, index, loss, optimizer, tt_splits=[0.5],
                      H=[1], D=[4], activations=['softplus'],
                      a=[2], b=[4], epochs=5000, batch_size=128):
    print_progress('reading data')
    read_data = read_fingerprints(filename, index)
    n_samples = len(read_data[0])

    lines = ['split,H,D,Activation,a,b;\n']
    for (tt_split, n_hlayers, dense_units,
         activation, n_pair_params,
         n_triplet_params) in list(product(tt_splits, H, D,
                                           activations,
                                           a, b)):
        global stage
        stage = 1
        split_index = int(np.round(tt_split * n_samples))

        tag = '{},{}-{},{}-{},{}'.format(activation, n_hlayers, dense_units,
                                         n_pair_params, n_triplet_params,
                                         tt_split)

        print_progress('preprocessing fingerprints'+tag)
        preprocessed_data = preprocess(read_data,
                                       n_pair_params, n_triplet_params,
                                       split_index)
        print_progress('training model: '+tag)
        loss_hist = train(preprocessed_data,
                          n_hlayers, dense_units, activation,
                          epochs, batch_size, loss, optimizer, tag)
        loss_hist_rescaled = loss_hist[::int(np.ceil(len(loss_hist)/1024))]
        #print('\n{0:.4f}'.format(min_loss))
        lines.append(tag.replace('-', ',')+';\n')
        line = ','.join(['{}'.format(loss_val) for
                         loss_val in loss_hist_rescaled])+';\n'
        lines.append(line)
    print_progress('writing loss histories')
    with open(filename.replace('.hdf5', '.csv'), 'w') as log:
        log.writelines(lines)


def preprocess(read_data,
               n_pair_params, n_trip_params, split_index):
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
    np.random.shuffle(padded)
    padded = padded.transpose([1, 0, 2])
    # print('Padded data shape:', padded.shape)
    # now (atom index, structure index, parameter set)

    assert not np.any(np.isnan(padded))
    assert not np.any(np.isinf(padded))

    train_inputs = padded[:, :split_index, :]
    valid_inputs = padded[:, split_index:, :]

    outputs_master = energy_list
    train_outputs = np.asarray(outputs_master[:split_index])
    valid_outputs = np.asarray(outputs_master[split_index:])
    return (train_inputs, valid_inputs,
            train_outputs, valid_outputs,
            sys_elements, max_per_element)


def train(data, n_hlayers, dense_units, activation,
          epochs, batch_size, loss, optimizer, filename):
    (train_inputs, valid_inputs,
     train_outputs, valid_outputs,
     sys_elements, max_per_element) = data

    global layers_id
    layers_id = 0
    input_vector_length = train_inputs.shape[-1]

    element_models = [element_model(input_vector_length, n_hlayers,
                                    dense_units, specie,
                                    activation=activation)
                      for specie in sys_elements]

    model = structure_model(input_vector_length, max_per_element,
                            sys_elements, element_models)
    model.compile(loss=loss, optimizer=optimizer, metrics=[mean_pred])

    a = model.predict([atom for atom in train_inputs])
    assert not np.any(np.isnan(a))
    assert not np.any(np.isinf(a))

    train_inputs = [atom for atom in train_inputs]
    valid_inputs = [atom for atom in valid_inputs]

    checkpointer = ModelCheckpoint(filepath=filename,
                                   verbose=0,
                                   monitor='val_loss',
                                   save_weights_only=False,
                                   period=100)

    es = EarlyStopping(min_delta=0.0001, patience=100)

    history = LossHistory()
    # nan_check = TerminateOnNaN()
    try:
        model.fit(train_inputs, train_outputs,
                  epochs=epochs, verbose=0,
                  validation_data=(valid_inputs, valid_outputs),
                  batch_size=batch_size, callbacks=[checkpointer,  # nan_check,
                                                    es, history, ])
    except (KeyboardInterrupt, SystemExit):
        print('\n')
        pass

    return history.losses

    # model.save_weights('my_model_weights.h5')
    # print(model.get_weights())


if __name__ == '__main__':
    filename = [fil for fil in os.listdir('.')
                if 'fingerprints' in fil and '.hdf5' in fil][0]
    grid_search_train(filename, reading_index, rmse_loss, optimizer,
                      tt_splits=set_tt_split, H=set_hlayers,
                      D=set_dense_units, a=set_pair_params,
                      b=set_trip_params, activations=set_activation,
                      epochs=epochs, batch_size=batch_size)
