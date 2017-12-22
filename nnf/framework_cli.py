"""
This module provides a command-line level argument parser for the framework.

Representation Workflow:
1) Read and Collate Structures
    parse_property: read property values, save to intermediate .hdf5
    collate_structures: read coordinate and element data,
                        save to intermediate .hdf5
2) Fingerprint Structures
    apply_descriptors: read intermediate .hdf5, process into fingerprints,
                       write to another .hdf5

Neural Network Workflow:
1) Load Fingerprints from HDF5
    ...

2) [under construction]
"""
import os
import argparse
import time

from nnf.io_structures import parse_property, collate_structures
from nnf.io_fingerprints import read_parameters_from_file, apply_descriptors
from nnf.io_utils import validate_hdf5
from configparser import ConfigParser


def initialize_argparser():
    """

    Initializes ArgumentParser for command-line operation.

    """
    argparser = argparse.ArgumentParser(description='Converts structures to \
                                        symmetry functions.')
    argparser.add_argument('action', choices=['fingerprint', 'collate',
                                              'train','validate'])
    argparser.add_argument('settings_file',
                           help='Filename of settings.')
    argparser.add_argument('--verbose', '-v', action='count')
    return argparser


class Settings:
    def __init__(self, action):
        self.definition_parser = ConfigParser()
        self.definition_parser.read("definitions.cfg")
        self.types = {}
        for option, value in self.definition_parser.items(action):
            self.types.setdefault(value, []).append(option)
        self.parser = ConfigParser()
        self.section = self.parser[action]
        self.settings = {}
        self.read('defaults.cfg')

    def read(self, filename):
        self.parser.read(filename)
        for key in self.types.get('str', []):
            self.settings[key] = self.section.get(key)
        for key in self.types.get('int', []):
            self.settings[key] = self.section.getint(key)
        for key in self.types.get('float', []):
            self.settings[key] = self.section.getfloat(key)
        for key in self.types.get('bool', []):
            self.settings[key] = self.section.getboolean(key)
        for key in self.types.get('strs', []):
            self.settings[key] = self.section.get(key).split(',')
        for key in self.types.get('ints', []):
            self.settings[key] = [int(val)
                                  for val in self.section.get(key).split(',')]
        for key in self.types.get('floats', []):
            self.settings[key] = [float(val)
                                  for val in self.section.get(key).split(',')]
        for key in self.types.get('bools', []):
            self.settings[key] = [bool(val)
                                  for val in self.section.get(key).split(',')]
        return self.settings


if __name__ == '__main__':
    args = initialize_argparser().parse_args()
    action = args.action
    settings_file = args.settings_file
    verbosity = args.verbosity

    settings = Settings(action).read(settings_file)
    input_name = settings['inputs_file']
    index = settings['index']
    output_name = settings['outputs_file']
    sys_elements = settings['sys_elements']

    if not len(sys_elements) == len(set(sys_elements)):
        raise ValueError('duplicate elements: ' + sys_elements)

    if ':' not in settings['index']:
        raise ValueError('invalid slice specified: ' + args.index)

    global t0
    t0 = time.time()

    if args.action == 'fingerprint':
        parameters = read_parameters_from_file(settings['parameters_file'])
        descriptor = settings['descriptor']
        derivs = settings['derivatives']
        apply_descriptors(input_name, output_name, sys_elements, parameters,
                          index=index, descriptor=descriptor,
                          derivs=derivs)
        validate_hdf5(output_name, sys_elements=sys_elements)

    elif args.action == 'collate':
        loose = os.path.isdir(input_name)
        energies_file = settings['energies_file']
        form = settings['input_format']
        keyword = settings['keyword']
        with open(energies_file, 'r') as propfile:
            proptext = propfile.read()
        energy_data = parse_property(proptext, loose=loose,
                                     keyword=keyword,
                                     index=index)
        # dictionary if loose, list otherwise
        collate_structures(input_name, output_name, sys_elements,
                           energy_data, index=index, form=form)
        validate_hdf5(output_name, sys_elements=sys_elements)

    elif args.action == 'validate':
        validate_hdf5(input_name, sys_elements=sys_elements)

    print(time.time() - t0, 'seconds elapsed.')
