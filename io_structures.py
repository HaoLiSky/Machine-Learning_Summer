"""

This module provides functions to read crystal/molecule data from files or
filetrees (using ASE) and process data into fingerprints using descriptors.

Representation Workflow:
parse_property: read property values, save to intermediate .hdf5
structures_to_hdf5: read coordinate and element data,
                     save to intermediate .hdf5
apply_fingerprints: read intermediate .hdf5, process into fingerprints,
                     write to another .hdf5
validate_hdf5: Check .hdf5 integrity for structures or fingerprints.

"""
import os
import argparse
import time
import re
import h5py
import traceback

import numpy as np
from fingerprints import bp_fingerprint, dummy_fingerprint
from itertools import islice
from ase.io import iread


def initialize_argparser():
    """

    Initializes ArgumentParser for command-line operation.

    """
    argparser = argparse.ArgumentParser(description='Converts structures to \
                                        symmetry functions.')
    argparser.add_argument('action', choices=['represent', 'parse',
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


def strslice(string):
    """

    Adapted from ase.io.formats.py to parse string into slice.

    Args:
        string (str): Slice as string.

    Returns:
        slice (slice): Slice.

    """
    if ':' not in string:
        return int(string)
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)


def slice_generator(generator, string):
    """

    Applies slicing to a generator.

    Args:
        generator: The generator to convert.
        string (str): Slice as string.

    Returns:
        An itertools.islice iterator.

    """
    if ':' not in string:
        return int(string)
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return islice(generator, *i)


def read_parameters(params_file):
    """
    Read parameter sets from file.

    Args:
        para_file (str): Parameters filename.

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


def parse_property(data, loose=False, keyword=None, index=':'):
    """

    Parses property values from file. Uses regex for flexible searching.
    If crystal/molecule files are loose (i.e. not in a combined file),
    property values must be specified as filename-value pairs.
    (e.g. CONTCAR_0: -0.4E+3; POSCAR_1: -0.3E+2 ...)
    If crystal/molecule input is one fine, values may be parsed with a
    specified *preceding* keyword/keystring.

    If loose,
        Each Filename & value must be joined with a colon, comma,
        or equals-sign (whitespace optional). Pairs may be comma-separated,
        colon-separated, and/or white-space-separated.

    If a keyword is given,
        Each keyword & value may be joined by whitespace, comma, colon,
        or equals-sign, and must be followed by a comma, a semicolon,
        whitespace, or a linebreak.

    Otherwise,
        Values may be separated by whitespace, linebreaks, commas,
        or semicolons.

    Args:
        data (str): Property data text.
        loose: True if crystal/molecule input is a directory.
        keyword (str): Optional string to find property values.
            (e.g "energy" to find "energy: 0.423eV"
            or "E0" to find "E0= -0.52E+02")
        index (str): Slice. Must match in parse_ase.

    Returns:
        energy_list: Property values as floats or dictionary of
            filename-value pairs if loose.

    """
    data = data + ';'  # ensures final value is not skipped
    if loose:
        parser = re.compile('([\w\._]+)(?:[=\s,:]+)(\S+)(?:[;,\s])')
        data_pairs = parser.findall(data)
        energy_list = {k: v for k, v in data_pairs}
        # full dictionary, one value per structure file
        return energy_list
    else:
        if keyword:
            parser = re.compile('(?:' + keyword
                                + '[=\s,:]*)([^;,\s]+)')
            energy_list = [float(match) for match in parser.findall(data)]

        else:
            parser = re.compile('([^,;\s]+)(?:[,;\s])')
            energy_list = [float(match) for match in parser.findall(data)]
            # list of values after slicing
            # assumed to match with ase.io.iread's slicing
        return energy_list[strslice(index)]


def write_structure(h5f, structure, sys_elements, symbol_order, s_name,
                    energy_val):
    """

    Writes one crystal/molecule object's data to .hdf5 file.

    Args:
        h5f: h5py object for writing.
        structure: ASE Atoms object.
        sys_elements: List of system-wide unique element names as strings.
        symbol_order: Dictionary of elements and specified order.
        s_name (str): Atoms' identifier to use as group name.
        energy_val (float): Atoms' corresponding property value.

    """
    coords = structure.get_positions(wrap=False)
    natoms = len(coords)
    species = structure.get_chemical_symbols()
    symbol_set = list(set(species))
    assert set(species).issubset(set(sys_elements))
    species_counts = np.asarray([species.count(cspecie)
                                 for cspecie
                                 in sys_elements]).astype('i4')
    assert sum(species_counts) == natoms
    species, coords = zip(*sorted(zip(species, coords),
                                  key=lambda x: symbol_order.get(x[0])))
    try:
        unit = structure.get_cell(complete=False)
        periodic = True
    except (ValueError, RuntimeError, AttributeError):
        unit = np.zeros((3, 3))
        periodic = False
    dset_name = 'structures/{}/coordinates'.format(s_name)
    dset_coords = h5f.create_dataset(dset_name,
                                     (natoms, 3),
                                     data=coords, dtype='f4',
                                     compression="gzip")
    dset_coords.attrs['natoms'] = natoms
    dset_coords.attrs['symbol_set'] = np.string_(symbol_set)
    dset_coords.attrs['species_counts'] = species_counts
    dset_coords.attrs['unit'] = unit
    dset_coords.attrs['periodic'] = periodic
    dset_coords.attrs['energy'] = energy_val


def structures_to_hdf5(input_name, output_name, sys_elements,
                       energy_data, index=':', form=None):
    """

    Parses crystal/molecule data and property data and writes to .hdf5 file.
    Data is stored in one group per crystal/molecule, named by the parent
    filename and the id of the crystal/molecule, assigned in order of
    writing.

    Example: the 31st dataset in the h5py object "h5f", corresponding to a
        molecule parsed from "au55.xyz", is stored in the .hdf5 location
        "h5f/structures/au55xyz_31/coordinates."

    Args:
        input_name (str): Filename or root directory of filetree of
            crystal/molecule data.
        output_name (str): Filename for .hdf5.
        index (str): Slice. Must match in parse_property.
        sys_elements: List of system-wide unique element names as strings.
        form (str): Optional file format string to pass to ASE's read function.
        energy_data: List of property values as floats or
            dictionary of filename-value pairs if loose.

    """
    prefix = input_name.replace('.', '') + '_'

    symbol_order = {k: v for v, k in enumerate(sys_elements)}
    with h5py.File(output_name, 'w', libver='latest') as h5f:
        sys_params = h5f.require_group('system')
        sys_params.attrs['sys_elements'] = np.string_(sys_elements)
        s_count = len(h5f.require_group('structures').keys())
        s_count_start = s_count + 0
        print(str(s_count_start), 'structures to start')
        loose = os.path.isdir(input_name)
        if loose:
            for root, dirs, files in slice_generator(os.walk(input_name),
                                                     index):
                for filename in files:
                    # print(os.path.join(root, filename))
                    try:
                        structures = iread(os.path.join(root, filename),
                                           format=form)
                    except (ValueError, IOError, IndexError):
                        continue
                    for structure in structures:
                        s_name = filename.replace('.', '') + '_' + str(s_count)
                        try:
                            energy_val = energy_data[filename]
                            write_structure(h5f, structure, sys_elements,
                                            symbol_order, s_name,
                                            energy_val)
                            s_count += 1
                            print('read', s_count, end='\r')
                        except KeyError:
                            print('no corresponding energy_data:', s_name)
                            continue
                        except AssertionError:
                            print('error in', s_name, end='\n\n')
                            traceback.print_exc()
                            continue
        else:
            structures = iread(input_name, format=form, index=index)
            for structure in structures:
                s_name = prefix + str(s_count)
                try:
                    energy_val = energy_data[s_count]
                    write_structure(h5f, structure, sys_elements,
                                    symbol_order, s_name, energy_val)
                    s_count += 1
                except IndexError:
                    print('no corresponding energy_data:', s_name)
                    continue
                except AssertionError:
                    print('error in', s_name, end='\n\n')
                    traceback.print_exc()
                    continue
                print('read', s_count, end='\r')
        print(str(s_count - s_count_start), 'structures parsed into .hdf5')


def read_structure(h5i, s_name):
    """

    Reads data for one crystal/molecule and correpsonding property data
    from .hdf5 file.

    Args:
        h5i : h5py object for reading.
        s_name (str): Group name in h5i corresponding to Atoms' identifier.

    Returns:
        coords (n x 3 numpy array): Coordinates for each atom.
        symbol_set: List of unique element names in structure as strings.
        species_counts: List of occurences for each element type.
        species_list: List of element namestring for each atom.
        unit (3x3 numpy array): Atoms' unit cell vectors (zeros if molecule).
        periodic: True if crystal.
        energy_val (float): Corresponding property value.

    """
    dset = h5i['structures'][s_name]['coordinates']
    coords = dset[()]
    symbol_set = [symbol.decode('utf-8')
                  for symbol in dset.attrs['symbol_set']]
    species_counts = dset.attrs['species_counts']
    assert len(symbol_set) == len(species_counts)
    species_list = []
    for symbol, ccount in zip(symbol_set, species_counts):
        species_list += [symbol] * ccount
    assert len(species_list) == len(coords)
    unit = dset.attrs['unit']
    periodic = dset.attrs['periodic']
    energy_val = dset.attrs['energy']

    return [coords, symbol_set, species_counts,
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
    elif descriptor == 'BP':
        inputs = bp_fingerprint(s_data, parameters, sys_elements)

    (coords, symbol_set, species_counts,
     species_list, unit, periodic, energy_val) = s_data

    for label, term in inputs:
        dname = 'structures/{}/{}'.format(s_name, label)
        h5o.create_dataset(dname, term.shape,
                           data=term, dtype='f4', compression="gzip")
    h5o['structures'][s_name].attrs['natoms'] = len(coords)
    h5o['structures'][s_name].attrs['symbol_set'] = np.string_(symbol_set)
    h5o['structures'][s_name].attrs['species_counts'] = species_counts
    h5o['structures'][s_name].attrs['energy'] = energy_val


def apply_descriptors(input_name, output_name, sys_elements, parameters,
                      descriptor='PH'):
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
            s_names = list(h5i.require_group('structures').keys())
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names):
                print('processing', str(j + 1).rjust(10), '/',
                      str(s_tot).rjust(10), end='\r')
                try:
                    s_data = read_structure(h5i, s_name)
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
                    'energy': 0,
                    'symbol_set': [],
                    'species_counts': [0],
                    'unit': [],
                    'periodic': False}
    sys_elements = [np.string_(x) for x in sys_elements]
    comp_coeff_list = []
    with h5py.File(filename, 'r', libver='latest') as h5f:
        s_names = list(h5f.require_group('structures').keys())
        n_structures = 0
        n_fingerprints = 0
        n_other = 0
        s_dset = h5f['structures']
        try:
            sys_elements = h5f['system'].attrs['sys_elements']
            p_dset = h5f['system']
            p_dims = [p_dset['pair'][()].shape[0],
                      p_dset['triplet'][()].shape[0]]
            print(p_dims)
        except KeyError:
            p_dims = []

        print(sys_elements)

        for j, s_name in enumerate(s_names):
            if 'coordinates' in s_dset[s_name].keys():
                dset = s_dset[s_name]['coordinates']
                keys_present = list(dset.attrs.keys())
                keys_needed = ['natoms', 'energy', 'symbol_set',
                               'species_counts', 'unit', 'periodic']
                missing_keys = set(keys_needed).difference(set(keys_present))

                data = {}
                for key in keys_needed:
                    if key in missing_keys:
                        data[key] = default_dict[key]
                    else:
                        data[key] = dset.attrs[key]

                assertions = [data['natoms'] == sum(data['species_counts']),
                              (set(data['symbol_set'])
                               .issubset(set(sys_elements))),
                              (len(data['species_counts'])
                               == len(sys_elements)),
                              data['unit'].shape == (3, 3)]
                warnings = ['natoms and species_counts mismatch :'
                            + str(data['natoms']) + ' ; '
                            + str(sum(data['species_counts'])),
                            'invalid elements ' + str(data['symbol_set']),
                            'species counts mismatch '
                            + str(len(data['species_counts'])),
                            'unit cell bad shape: ' + str(data['unit'].shape)]
                if all(assertions):
                    n_structures += 1
                    comp_coeff_list.append(data['species_counts'])
                else:
                    print(s_name)
                    print('missing attributes: ', ','.join(missing_keys))
                    for k, err in enumerate(warnings):
                        if not assertions[k]:
                            print(err)
            elif 'BP_g1' in s_dset[s_name].keys():
                dset = s_dset[s_name]
                keys_present = list(dset.attrs.keys())
                keys_needed = ['natoms', 'energy', 'symbol_set',
                               'species_counts']
                missing_keys = set(keys_needed).difference(set(keys_present))

                data = {}
                for key in keys_needed:
                    if key in missing_keys:
                        data[key] = default_dict[key]
                    else:
                        data[key] = dset.attrs[key]
                Expected_shape

                assertions = [data['natoms'] == sum(data['species_counts']),
                              (set(data['symbol_set'])
                               .issubset(set(sys_elements))),
                              ((len(data['species_counts']) == len(
                               sys_elements))),
                              ([dset['BP_g1'][()].shape[-1],
                                dset['BP_g2'][()].shape[-1]]) == p_dims]
                warnings = ['natoms and species_counts mismatch :'
                            + str(data['natoms']) + ' ; '
                            + str(sum(data['species_counts'])),
                            'invalid elements ' + str(
                                data['symbol_set']),
                            'species counts mismatch '
                            + str(data['species_counts']),
                            'descriptor dimension mismatch.']
                if all(assertions):
                    comp_coeff_list.append(data['species_counts'])
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
        comp_coeffs = np.asarray(comp_coeff_list)
        max_each = np.amax(comp_coeffs, axis=0)
        sums = np.sum(comp_coeffs, axis=1)
        largest_struct = comp_coeffs[np.argmax(sums), :]
        smallest_struct = comp_coeffs[np.argmin(sums), :]
        print('Maximum atoms of each element type:', max_each)
        print('Largest structure:', largest_struct)
        print('Smallest Structure:', smallest_struct)
        print('\n\n----\n', n_structures, n_fingerprints, n_other)
        return n_structures, n_fingerprints, n_other


if __name__ == '__main__':
    argparser = initialize_argparser()
    args = argparser.parse_args()

    sys_elements = args.sys_elements.split(',')
    if not len(sys_elements) == len(set(list(sys_elements))):
        raise ValueError('duplicate elements in symbol_set: ' + sys_elements)

    if ':' not in args.index:
        raise ValueError('invalid slice specified: ' + args.index)

    input_name = args.input

    global t0
    t0 = time.time()

    if args.action == 'represent':
        if args.output:
            output_name = (os.path.splitext(args.output)[0]
                           + '_fingerprints.hdf5')
        else:
            output_name = (os.path.splitext(input_name)[0]
                           .replace('_structures', '')
                           + '_fingerprints.hdf5')
        parameters = read_parameters(args.input2)
        apply_descriptors(input_name, output_name, sys_elements, parameters,
                          descriptor=args.descriptor)

    elif args.action == 'parse':
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
        structures_to_hdf5(input_name, output_name, sys_elements,
                           energy_data, index=args.index, form=args.form)
    elif args.action == 'validate':
        validate_hdf5(input_name, sys_elements=sys_elements)
    print(time.time() - t0, 'seconds elapsed.')
