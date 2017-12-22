"""

"""
import os
import numpy as np
from time import time
from datetime import datetime
from itertools import product
from keras.models import Model
from keras.layers import Input, Dense, Add, Dropout, Multiply
from keras.layers import ActivityRegularization
from keras.optimizers import Nadam, Adadelta, Adam, SGD
from keras.callbacks import ModelCheckpoint
from keras.callbacks import TerminateOnNaN

from nnf.io_utils import slice_from_str
from nnf.custom_keras_utils import spread, mean_pred, LossHistory
from nnf.nn_startup import preprocess, read_preprocessed

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

reading_index = slice(None, None, None)

optimizer_dict = {'adam': Adam(),
                  'nadam': Nadam(),
                  'sgd': SGD(lr=0.0000001, decay=0,
                             momentum=0.6, nesterov=True),
                  'adadelta': Adadelta(clipnorm=1.)}

# arbitrary fixed seed for reproducibility
np.random.seed(8)

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


def grid_search(loss, settings, restart=False):
    filename = settings['inputs_file']
    index = slice_from_str(settings['index'])
    epochs = settings['epochs']

    hyperparameter_sets = [settings['batch_size'],
                           settings['optimizer'],
                           settings['activation'],
                           settings['H'],
                           settings['D'],
                           settings['l1'],
                           settings['l2'],
                           settings['dropout'],
                           settings['pair_sets'],
                           settings['triplet_sets'],
                           settings['test_chunk']]

    for (batch_size, optimizer, activation,
         n_hlayers, dense_units,
         l1_reg, l2_reg, dropout,
         n_pair_params, n_triplet_params,
         k_test_ind) in list(product(*hyperparameter_sets)):
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
        print_progress('wrapping up: ' + tag)


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