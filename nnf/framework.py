"""
Command-line level argument parser for the framework.

1) Collate Molecules

2) Make Fingerprints

3) Preprocess Fingerprints

4) Train using Keras model(s)

5) Predict properties (e.g. energy) using trained model(s)
"""
import os
import argparse
import time
from nnf.batch_collate import BatchCollator
from nnf.batch_fingerprint import FingerprintProcessor
from nnf.batch_preprocess import DataPreprocessor
from nnf.io_utils import SettingsParser
from nnf.network import Network

PACKAGE_PATH = os.path.dirname(__file__)


def initialize_argparser():
    """
    Initializes ArgumentParser for command-line operation.
    """
    description = 'Framework for fitting neural network potentials.'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('action', choices=['Collate', 'Fingerprint',
                                              'Preprocess', 'Network'])
    argparser.add_argument('--settings_file', '-s', default='settings.cfg',
                           help='Filename of settings.')
    argparser.add_argument('--verbosity', '-v', default=0,
                           action='count')
    argparser.add_argument('--export', '-e', action='store_true',
                           help='Export entries to csv.')
    argparser.add_argument('--GridSearch', '-g', action='store_true',
                           help='Initialized grid search with network.')
    argparser.add_argument('--plot', '-p', action='store_true',
                           help='Plot and Save figures.')
    argparser.add_argument('--force', '-f', action='store_true',
                           help='Force overwrite/merge. (prompt otherwise)')
    args = argparser.parse_args()
    return args


if __name__ == '__main__':
    args = initialize_argparser()
    action = args.action
    settings = SettingsParser(action).read(args.settings_file)
    settings['verbosity'] = args.verbosity

    inputs_name = settings['inputs_name']
    outputs_name = settings['outputs_name']
    force = args.force
    if os.path.isfile(outputs_name) and not force:
        while True:
            reply = str(input('File "{}"exists. \
            Merge/overwrite?'.format(outputs_name)).lower().strip())
            if reply[0] == 'y':
                break
            elif reply[0] == 'n':
                try:
                    output_split = outputs_name.split('Copy')
                    ind = int(output_split[-1]) + 1
                    outputs_name = '{}Copy{}'.format(output_split[0], ind)
                except (IndexError, ValueError):
                    outputs_name = '{}Copy{}'.format(outputs_name, 1)
                break
    t0 = time.time()

    if action == 'Collate':
        # Parse molecule/crystal data into one .hdf5 file
        energies_file = settings['energies_file']
        sys_elements = settings['sys_elements']
        assert sys_elements != ['None']
        assert len(sys_elements) == len(set(sys_elements))
        assert ':' in settings['index']
        collator = BatchCollator(outputs_name, settings, sys_elements)
        if os.path.isfile(inputs_name):
            collator.parse_molecules(inputs_name, energies_file)
        else:
            collator.parse_loose_molecules(inputs_name, energies_file)
        if args.export:
            collator.export_entries(inputs_name.split('.')[0] + '.csv')
    elif action == 'Fingerprint':
        # Create fingerprints from collated .hdf5 file
        parameters_file = settings['parameters_file']
        sys_elements = settings['sys_elements']
        assert sys_elements != ['None']
        assert len(sys_elements) == len(set(sys_elements))
        assert ':' in settings['index']
        processor = FingerprintProcessor(outputs_name, parameters_file,
                                         settings)
        processor.process_collated(inputs_name)
        if args.export:
            processor.export_entries(inputs_name.split('.')[0] + '.csv')
    elif action == 'Preprocess':
        # Preprocess fingerprints into normalized vectors for machine learning
        partitions_file = settings['partitions_file']
        split_fraction = settings['split_fraction']
        kfold = settings['kfold']
        simple = settings['simple']
        sys_elements = settings['sys_elements']
        assert sys_elements != ['None']
        assert len(sys_elements) == len(set(sys_elements))
        assert ':' in settings['index']
        preprocessor = DataPreprocessor(settings)
        preprocessor.read_fingerprints(inputs_name)
        subdivisions = [int(val) for val in settings['subdivisions']]
        preprocessor.subdivide_by_parameter_set(subdivisions)
        preprocessor.preprocess_fingerprints()
        preprocessor.to_file(outputs_name)
        if simple:
            preprocessor.generate_simple_partitions(partitions_file,
                                                    split_fraction,
                                                    k=kfold)
        else:
            preprocessor.generate_partitions(partitions_file, split_fraction,
                                             k=kfold)
    elif action == 'Network':
        # Create and train Keras models
        partitions_file = settings['partitions_file']
        network = Network(settings)
        # Combinatorial grid-search for parameter/hyperparameter space
        if args.GridSearch:
            network.grid_search(args.settings_file, outputs_name)
        else:
            network.load_data(inputs_name, partitions_file, settings)
            final_loss = network.train_network(settings)
            print('\n\nFinal loss:', final_loss)
    print('\n\n{}'.format(time.time() - t0), 'seconds elapsed.')
