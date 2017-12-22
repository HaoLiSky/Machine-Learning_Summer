"""
This module provides input/output functions for fingerprints
using descriptors.
"""
import h5py
import traceback

import numpy as np
from nnf.fingerprints import bp_fingerprint, dummy_fingerprint
from nnf.io_utils import read_from_group, write_to_group
from nnf.io_structures import read_collated_structure, slice_from_str


def apply_descriptors(input_name, output_name, sys_elements, parameters,
                      descriptor='dummy', index=':', derivs=False):
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
        index (str): Slice. Defaults to ':' for all entries.
        derivs (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.

    """

    with h5py.File(input_name, 'r', libver='latest') as h5i:
        s_names = [name.decode('utf-8')
                   for name in h5i['system']['s_names_list'][()]]
        with h5py.File(output_name, 'a', libver='latest') as h5o:
            sys_entries = []
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names[slice_from_str(index)]):
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
                                              descriptor=descriptor,
                                              derivs=derivs)
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
                           {'pair_params'   : parameters[0],
                            'triplet_params': parameters[1]})

            write_to_group(h5o, 'system',
                           {},
                           {'sys_entries' : sys_entries,
                            's_names_list': s_names_list},
                           dict_dset_types={'s_names_list': s_names_list.dtype,
                                            'sys_entries' : sys_entries.dtype},
                           maxshape=(None,))


def load_fingerprints_from_file(filename, sys_elements, indexing=':',
                                pad=False, per_structure=False):
    """

    Top-level function to load fingerprints and corresponding structure
    attributes from .hdf5 file into numpy arrays.

    Args:
        filename (str): .hdf5 filename.
        sys_elements: List of system-wide unique element names as strings.
        indexing (str): Slice. Defaults to ':' for all entries.
        pad: Whether to pad by elements to produce equal-length fingerprints
            per type.
        per_structure: Whether to return of list of (G_1, G_2) for each
            structure or list of all G1 & list of all G2

    Returns:
        List of fingerprints (either per structure or per type)
        and list of element counts per structure.

    """
    with h5py.File(filename, 'r', libver='latest') as h5f:
        s_names = [line.split(b';')[0].decode('utf-8')
                   for line in h5f['system']['sys_entries'][()]]
        fingerprint_set_list = []
        element_counts_per_structure = []
        energies = []
        for j, s_name in enumerate(s_names[slice_from_str(indexing)]):
            [dsets_dict, natoms,
             element_set, element_counts,
             elements_list, energy_val] = read_fingerprint(h5f,
                                                           s_name,
                                                           sys_elements)
            element_counts_per_structure.append(element_counts)
            energies.append(energy_val)
            fingerprint_set_list.append(list(dsets_dict.values()))
    max_element_counts = np.amax(element_counts_per_structure, axis=0)
    if pad:
        fingerprints = pad_fingerprint_by_element(fingerprint_set_list,
                                                  element_counts_per_structure,
                                                  max_element_counts)
    else:
        fingerprints = fingerprint_set_list

    if per_structure:
        return fingerprints, element_counts_per_structure
    else:
        g1_list, g2_list = zip(*fingerprints)
        return g1_list, g2_list, element_counts_per_structure


def make_fingerprint(h5f, s_data, s_name, parameters,
                     sys_elements, descriptor='dummy',
                     derivs=False):
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
        derivs (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.

    """
    inputs = []
    shapes = []
    if descriptor == 'dummy':
        inputs, shapes = dummy_fingerprint(s_data, parameters,
                                           sys_elements)
    elif descriptor == 'bp':
        inputs, shapes = bp_fingerprint(s_data, parameters, sys_elements,
                                        derivs=derivs)

    (coords, element_set, element_counts,
     element_list, unit, periodic, energy_val) = s_data

    dict_dsets = {label: term for label, term in inputs}
    dict_attrs = {'natoms'        : len(coords),
                  'element_set'   : np.string_(element_set),
                  'element_counts': element_counts,
                  'energy'        : energy_val}
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
    grouppath = 'structures/{}'.format(s_name)
    attrs_dict, dsets_dict = read_from_group(h5f,
                                             grouppath)

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


def pad_fingerprint_by_element(input_data, compositions, final_layers):
    """

    Slice fingerprint arrays by elemental composition and
    arrange slices onto empty arrays to produce sets of padded fingerprints
    (i.e. equal length for each type of fingerprint)

    Args:
        input_data: List of lists of
            [g1(n, ...), g2(n, ...), ... ] where n = sum(comp)
        compositions: list of "layer" heights (per element) in # of atoms
        final_layers: list of desired "layer" heights (per element)
            in # of atoms
    Returns:
        output_data: List of lists of
            [g1(N, ...), g2(N, ...), ... ] where N = sum(max_per_element)

    """
    new_input_data = []
    for structure, initial_layers in zip(input_data, compositions):
        assert len(final_layers) == len(initial_layers)
        new_fingerprints = []
        for data in structure:  # e.g. G1, G2, ...
            data_shape = data.shape
            natoms_i = data_shape[0]
            natoms_f = sum(final_layers)
            natoms_diff = natoms_f - natoms_i
            secondary_dims = len(data_shape) - 1
            pad_widths = [(natoms_diff, 0)] + [(0, 0)] * secondary_dims
            # tuple of (header_pad, footer_pad) per dimension
            data_f = np.pad(np.ones(data.shape), pad_widths, 'edge') * -1

            n = len(initial_layers)
            slice_pos = [sum(initial_layers[:i + 1])
                         for i in range(n - 1)]
            # row indices to slice to create n sections
            data_sliced = np.split(data, slice_pos)

            start_pos = [sum(initial_layers[:i]) for i in range(n)]
            # row indices to place sections of correct length

            for sect, start in zip(data_sliced, start_pos):
                end = start + len(sect)
                data_f[start:end, ...] = sect
            new_fingerprints.append(np.asarray(data_f))
        new_input_data.append(new_fingerprints)
    return new_input_data
