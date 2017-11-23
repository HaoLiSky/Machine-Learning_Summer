"""
This module provides input/output functions for fingerprints
using descriptors.
"""
import h5py
import traceback

import numpy as np
from fingerprints import bp_fingerprint, dummy_fingerprint


def read_structure(h5i, s_name, sys_elements):
    """

    Reads data for one crystal/molecule and correpsonding property data
    from .hdf5 file.

    Args:
        h5i : h5py object for reading.
        s_name (str): Group name in h5i corresponding to Atoms' identifier.
        sys_elements: List of system-wide unique element names as strings.

    Returns:
        coords (n x 3 numpy array): Coordinates for each atom.
        element_set: List of unique element names in structure as strings.
        species_counts: List of occurences for each element type.
        species_list: List of element namestring for each atom.
        unit (3x3 numpy array): Atoms' unit cell vectors (zeros if molecule).
        periodic: True if crystal.
        energy_val (float): Corresponding property value.

    """
    dset = h5i['structures'][s_name]
    coords = dset['coordinates'][()]
    element_set = [symbol.decode('utf-8')
                   for symbol in dset.attrs['element_set']]
    species_counts = dset.attrs['species_counts']
    assert len(sys_elements) == len(species_counts)
    species_list = []
    for symbol, ccount in zip(sys_elements, species_counts):
        species_list += [symbol] * ccount
    assert len(species_list) == len(coords)
    unit = dset.attrs['unit']
    periodic = dset.attrs['periodic']
    energy_val = dset.attrs['energy']

    return [coords, element_set, species_counts,
            species_list, unit, periodic, energy_val]


def make_fingerprint(h5o, s_data, s_name, parameters,
                     sys_elements, descriptor='dummy'):
    """

    Reads data for one crystal/molecule and corresponding property data
    from .hdf5 file.

    Args:
        h5o: h5py object for writing.
        s_data: List of data (output of read_structure).
        s_name (str): Atoms' identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        sys_elements: List of system-wide unique element names as strings.
        descriptor: Descriptor to use.

    """
    if descriptor == 'dummy':
        inputs = dummy_fingerprint(s_data, parameters,
                                   sys_elements)
    elif descriptor == 'bp':
        inputs = bp_fingerprint(s_data, parameters, sys_elements)

    (coords, element_set, species_counts,
     species_list, unit, periodic, energy_val) = s_data

    for label, term in inputs:
        dname = 'structures/{}/{}'.format(s_name, label)
        h5o.create_dataset(dname, term.shape,
                           data=term, dtype='f4', compression="gzip")
    h5o['structures'][s_name].attrs['natoms'] = len(coords)
    h5o['structures'][s_name].attrs['element_set'] = np.string_(element_set)
    h5o['structures'][s_name].attrs['species_counts'] = species_counts
    h5o['structures'][s_name].attrs['energy'] = energy_val


def apply_descriptors(input_name, output_name, sys_elements, parameters,
                      descriptor='dummy'):
    """

    Reads crystal/molecule data from .hdf5, creates fingerprints, and writes to
    new .hdf5.

    Args:
        input_name (str): Filename of .hdf5 for reading crystal/molecule data.
        parameters : List of descriptor parameters.
        output_name (str): Filename of .hdf5 for writing fingerprint data.
        descriptor (str): Descriptor to use to represent crystal/molecule data.
        sys_elements: List of system-wide unique element names as strings.

    """

    with h5py.File(input_name, 'r', libver='latest') as h5i:
        with h5py.File(output_name, 'w', libver='latest') as h5o:
            h5o.create_dataset('system/pair',
                               parameters[0].shape,
                               data=parameters[0], dtype='f4',
                               compression="gzip")
            h5o.create_dataset('system/triplet',
                               parameters[1].shape,
                               data=parameters[1], dtype='f4',
                               compression="gzip")
            h5o['system'].attrs['sys_elements'] = np.string_(sys_elements)
            s_names = sorted(list(h5i.require_group('structures').keys()))
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names):
                print('processing', str(j + 1).rjust(10), '/',
                      str(s_tot).rjust(10), end='\r')
                try:
                    s_data = read_structure(h5i, s_name, sys_elements)
                    make_fingerprint(h5o, s_data, s_name, parameters,
                                     sys_elements, descriptor=descriptor)
                    s_count += 1
                except AssertionError:
                    print('error in', s_name, end='\n\n')
                    traceback.print_exc()
                    f_count += 1
                    continue
            print(str(s_count), 'fingerprints created')
            if f_count > 0:
                print(str(f_count), 'fingerprint(s) failed')
