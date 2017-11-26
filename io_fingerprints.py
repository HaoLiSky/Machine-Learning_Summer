"""
This module provides input/output functions for fingerprints
using descriptors.
"""
import h5py
import traceback

import numpy as np
from fingerprints import bp_fingerprint, dummy_fingerprint
from io_hdf5 import read_from_group, write_to_group
from io_structures import read_collated_structure


def apply_descriptors(input_name, output_name, sys_elements, parameters,
                      descriptor='dummy'):
    """

    Top-level function to create fingerprints from collated crystal/molecule
    data. Reads from input .hdf5, creates fingerprints, and writes to output
     .hdf5.

    Args:
        input_name (str): Filename of .hdf5 for reading crystal/molecule data.
        parameters: List of descriptor parameters.
        output_name (str): Filename of .hdf5 for writing fingerprint data.
        descriptor (str): Descriptor to use to represent crystal/molecule data.
        sys_elements: List of system-wide unique element names as strings.

    """

    with h5py.File(input_name, 'r', libver='latest') as h5i:
        s_names = [name.decode('utf-8')
                   for name in h5i['system']['s_names_list'][()]]
        with h5py.File(output_name, 'a', libver='latest') as h5o:
            sys_entries = []
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names):
                print('processing', str(j + 1).rjust(10), '/',
                      str(s_tot).rjust(10), end='\r')
                try:
                    s_data = read_collated_structure(h5i, s_name, sys_elements)
                    s_name_new = '{}_{}'.format('_'.join(s_name.
                                                         split('_')[:-1]),
                                                s_count)
                    f_data = make_fingerprint(h5o, s_data, s_name_new,
                                              parameters,
                                              sys_elements,
                                              descriptor=descriptor)
                    sys_entries.append(f_data)
                    s_count += 1
                except AssertionError:
                    print('error in', s_name, end='\n\n')
                    traceback.print_exc()
                    f_count += 1
                    continue
            print(str(s_count), 'new fingerprints created in', output_name)
            if f_count > 0:
                print(str(f_count), 'fingerprint(s) failed')

            s_names_list = np.asarray([np.string_(s_name)
                                       for s_name in s_names])
            sys_entries = np.asarray([b';'.join(entries)
                                      for entries in sys_entries])
            write_to_group(h5o, 'system',
                           {'sys_elements': np.string_(sys_elements)},
                           {'pair_params': parameters[0],
                            'triplet_params': parameters[1]})

            write_to_group(h5o, 'system',
                           {},
                           {'sys_entries': sys_entries,
                            's_names_list': s_names_list},
                           dict_dset_types={'s_names_list': s_names_list.dtype,
                                            'sys_entries': sys_entries.dtype},
                           maxshape=(None,))


def make_fingerprint(h5f, s_data, s_name, parameters,
                     sys_elements, descriptor='dummy'):
    """

    Reads data for one crystal/molecule and corresponding property data
    from .hdf5 file.

    Args:
        h5f: h5py object for writing.
        s_data: List of data (output of read_collated_structure).
        s_name (str): Atoms' identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        sys_elements: List of system-wide unique element names as strings.
        descriptor: Descriptor to use.

    """
    inputs = []
    shapes = []
    if descriptor == 'dummy':
        inputs, shapes = dummy_fingerprint(s_data, parameters,
                                           sys_elements)
    elif descriptor == 'bp':
        inputs, shapes = bp_fingerprint(s_data, parameters, sys_elements)

    (coords, element_set, element_counts,
     element_list, unit, periodic, energy_val) = s_data

    dict_dsets = {label: term for label, term in inputs}
    dict_attrs = {'natoms': len(coords),
                  'element_set': np.string_(element_set),
                  'element_counts': element_counts,
                  'energy': energy_val}
    group_name = 'structures/{}'.format(s_name)
    write_to_group(h5f, group_name, dict_attrs, dict_dsets)

    fingerprint_shapes = np.string_(','.join([str(shape)
                                              for shape in shapes]))
    f_data = [np.string_(s_name),
              np.string_(str(len(coords))),
              np.string_(','.join(element_set)),
              np.string_(','.join([str(x) for x in element_counts])),
              fingerprint_shapes,
              np.string_(energy_val)]

    return f_data


def read_fingerprint(h5f, s_name, sys_elements):
    """

    Reads fingerprints and attributes for one crystal/molecule
    and correpsonding property data from .hdf5 file.

    Args:
        h5f: h5py object for reading.
        s_name (str): Group name in h5f corresponding to Atoms' identifier.
        sys_elements: List of system-wide unique element names as strings.

    Returns:
        dsets_dict: Dictionary of fingerprint names and their ndarrays.
        natoms (int): Number of atoms in structure
        element_set: List of unique element names in structure as strings.
        elements_counts: List of occurences for each element type.
        elements_list: List of element namestring for each atom.
        energy_val (float): Corresponding property value.

    """
    attrs_dict, dsets_dict = read_from_group(h5f,
                                             'structures/{}'.format(s_name))

    natoms = attrs_dict['natoms']
    element_set = [symbol.decode('utf-8')
                   for symbol in attrs_dict['element_set']]
    element_counts = attrs_dict['element_counts']
    assert len(sys_elements) == len(element_counts)
    elements_list = []
    for symbol, ccount in zip(sys_elements, element_counts):
        elements_list += [symbol] * ccount
    assert len(elements_list) == natoms
    energy_val = attrs_dict['energy']

    return [dsets_dict, natoms, element_set, element_counts,
            elements_list, energy_val]


def read_parameters_from_file(params_file):
    """
    Read parameter sets from file.

    Args:
        params_file (str): Parameters filename.

    Returns:
        pairs: List of parameters for generating pair functions
        triplets: List of parameters for generating triplet functions

    """

    with open(params_file, 'r') as fil:
        lines = fil.read().splitlines()

    pairs, triplets = [], []
    for line in lines:
        if not line:
            continue
        entries = line.split()
        if entries[0][0] == 'p':
            pairs.append([float(para) for para in entries[1:]])
        elif entries[0][0] == 't':
            triplets.append([float(para) for para in entries[1:]])
        else:
            raise ValueError('invalid keyword!')

    return np.asarray(pairs), np.asarray(triplets)
