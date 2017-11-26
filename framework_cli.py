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
import h5py

import numpy as np
from io_structures import parse_property, collate_structures
from io_fingerprints import read_parameters_from_file, apply_descriptors


def initialize_argparser():
    """

    Initializes ArgumentParser for command-line operation.

    """
    argparser = argparse.ArgumentParser(description='Converts structures to \
                                        symmetry functions.')
    argparser.add_argument('action', choices=['fingerprint', 'collate',
                                              'validate'])
    argparser.add_argument('input',
                           help='.hdf5 for representation or ase-compatible \
                                 file or root of filetree for parsing')
    argparser.add_argument('input2',
                           help='filename of property data to parse \
                           or parameters to use in fingerprint')
    argparser.add_argument('sys_elements',
                           help='comma-separated list of unique elements; \
                           e.g. "Au" or "Ba,Ti,O"')
    argparser.add_argument('-o', '--output',
                           help='specify filename for .hdf5 output; \
                                default mirrors input name')
    argparser.add_argument('-k', '--keyword',
                           help='keyword to parse embedded property data; \
                           default: None')
    argparser.add_argument('-i', '--index', default=':',
                           help='slicing using numpy convention; \
                           e.g. ":250" or "-500:" or "::2"')
    argparser.add_argument('-f', '--form',
                           help='file format for ase structure/molecule \
                           parsing. If unspecified, ASE will guess.')
    argparser.add_argument('-d', '--descriptor', choices=['bp', 'dummy'],
                           default='dummy',
                           help='method of fingerprinting data; \
                           default: dummy (placeholder mimicking BP)')
    return argparser


def validate_hdf5(filename, sys_elements=[]):
    """

    Check .hdf5 integrity for structures or fingerprints.
    Presently, only checks for the presence of keys, not
    typing or dimensionality.

    Args:
        filename: Filename of .hdf5
        sys_elements: Optional argument, in case it cannot be read from
        the file.

    Returns:
        n_structures (int): Number of valid structures saved.
        n_fingerprints (int): Number of valid fingerprints saved.
        n_other (int): Number of other keys found in 'structures' group.

    """
    default_dict = {'natoms': 0,
                    'energy': '',
                    'element_set': [],
                    'element_counts': [0],
                    'unit_vectors': [],
                    'periodic': False}
    
    if not sys_elements:
        sys_elements = [np.string_(x) for x in sys_elements]
    comp_coeff_list = []
    with h5py.File(filename, 'r', libver='latest') as h5f:
        s_names = sorted(list(h5f.require_group('structures').keys()))
        n_structures = 0
        n_fingerprints = 0
        n_other = 0
        s_dset = h5f['structures']
        try:
            sys_elements = h5f['system'].attrs['sys_elements']
            p_dset = h5f['system']
            fingerprint_keys = sorted((set((p_dset.keys()))
                                      .difference({'system',
                                                   'sys_entries',
                                                   'coordinates',
                                                   's_names_list'})))
            p_dims = [p_dset[key][()].shape[0] for key in fingerprint_keys]
        except KeyError:
            p_dims = []

        for j, s_name in enumerate(s_names):
            if 'coordinates' in s_dset[s_name].keys():
                dset = s_dset[s_name]
                keys_present = list(dset.attrs.keys())
                keys_needed = ['natoms', 'energy', 'element_set',
                               'element_counts', 'unit_vectors', 'periodic']
                missing_keys = set(keys_needed).difference(set(keys_present))

                data = {}
                for key in keys_needed:
                    if key in missing_keys:
                        data[key] = default_dict[key]
                    else:
                        data[key] = dset.attrs[key]

                assertions = [data['natoms'] == sum(data['element_counts']),
                              (set(data['element_set'])
                               .issubset(set(sys_elements))),
                              (len(data['element_counts'])
                               == len(sys_elements)),
                              np.asarray(data['unit_vectors']).shape == (3, 3)]
                warnings = ['natoms and element_counts mismatch :'
                            + str(data['natoms']) + ' ; '
                            + str(sum(data['element_counts'])),
                            'invalid elements ' + str(data['element_set']),
                            'species counts mismatch '
                            + str(len(data['element_counts'])),
                            'unit cell bad shape: ' +
                            str(data['unit_vectors'])]
                if all(assertions):
                    n_structures += 1
                    comp_coeff_list.append(data['element_counts'])
                else:
                    print(s_name)
                    print('missing attributes: ', ','.join(missing_keys))
                    for k, err in enumerate(warnings):
                        if not assertions[k]:
                            print(err)
            elif list(s_dset[s_name].keys()):
                dset = s_dset[s_name]
                keys_present = list(dset.attrs.keys())
                keys_needed = ['natoms', 'energy', 'element_set',
                               'element_counts']
                missing_keys = set(keys_needed).difference(set(keys_present))

                data = {}
                for key in keys_needed:
                    if key in missing_keys:
                        data[key] = default_dict[key]
                    else:
                        data[key] = dset.attrs[key]

                assertions = [data['natoms'] == sum(data['element_counts']),
                              (set(data['element_set'])
                               .issubset(set(sys_elements))),
                              ((len(data['element_counts']) == len(
                               sys_elements))),
                              ([dset[fp][()].shape[-1] for fp in dset.keys()]
                               == p_dims)]
                warnings = ['natoms and element_counts mismatch :'
                            + str(data['natoms']) + ' ; '
                            + str(sum(data['element_counts'])),
                            'invalid elements ' + str(
                                data['element_set']),
                            'species counts mismatch '
                            + str(data['element_counts']),
                            'descriptor dimension mismatch: {}, {}'.format(
                                [dset[fp][()].shape[-1] for fp in dset.keys()],
                                p_dims)]
                if all(assertions):
                    comp_coeff_list.append(data['element_counts'])
                    n_fingerprints += 1
                else:
                    print(s_name)
                    print('missing attributes: ', ','.join(missing_keys))
                    for k, err in enumerate(warnings):
                        if not assertions[k]:
                            print(err)
            else:
                n_other += 1
                print('\n' + s_name, end=';\n')
            print(n_structures, n_fingerprints, n_other, end='\r')
        if comp_coeff_list:
            comp_coeffs = np.asarray(comp_coeff_list)
            max_each = np.amax(comp_coeffs, axis=0)
            sums = np.sum(comp_coeffs, axis=1)
            largest_struct = comp_coeffs[np.argmax(sums), :]
            smallest_struct = comp_coeffs[np.argmin(sums), :]
            print('Maximum atoms of each element type:', max_each)
            print('Largest structure:', largest_struct)
            print('Smallest Structure:', smallest_struct)
        print('\n----\n', n_structures, 'structures,',
              n_fingerprints, 'fingerprints,', n_other, 'other')
        return n_structures, n_fingerprints, n_other


if __name__ == '__main__':
    argparser = initialize_argparser()
    args = argparser.parse_args()

    sys_elements = args.sys_elements.split(',')
    if not len(sys_elements) == len(set(list(sys_elements))):
        raise ValueError('duplicate elements in element_set: ' + sys_elements)

    if ':' not in args.index:
        raise ValueError('invalid slice specified: ' + args.index)

    input_name = args.input

    global t0
    t0 = time.time()

    if args.action == 'fingerprint':
        if args.output:
            output_name = (os.path.splitext(args.output)[0]
                           + '.hdf5')
        else:
            output_name = (os.path.splitext(input_name)[0]
                           .replace('_structures', '')
                           + '_fingerprints.hdf5')
        parameters = read_parameters_from_file(args.input2)
        apply_descriptors(input_name, output_name, sys_elements, parameters,
                          descriptor=args.descriptor)
        validate_hdf5(output_name, sys_elements=sys_elements)

    elif args.action == 'collate':
        if args.output:
            output_name = (os.path.splitext(args.output)[0]
                           + '_structures.hdf5')
        else:
            output_name = (os.path.splitext(input_name)[0]
                           + '_structures.hdf5')

        loose = os.path.isdir(input_name)
        with open(args.input2, 'r') as propfile:
            proptext = propfile.read()
        energy_data = parse_property(proptext, loose=loose,
                                     keyword=args.keyword,
                                     index=args.index)
        # dictionary if loose, list otherwise
        collate_structures(input_name, output_name, sys_elements,
                           energy_data, index=args.index, form=args.form)
        validate_hdf5(output_name, sys_elements=sys_elements)
    elif args.action == 'validate':
        validate_hdf5(input_name, sys_elements=sys_elements)
    print(time.time() - t0, 'seconds elapsed.')
