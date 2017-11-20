"""

This module provides functions to read crystal/molecule data from files or
filetrees (using ASE) and process data into fingerprints using descriptors.

Representation Workflow:
parse_property : read property values, save to intermediate .hdf5
parse_ase : read coordinate and element data, save to intermediate .hdf5
apply_fingerprints : read intermediate .hdf5, process into fingerprints,
                     write to another .hdf5

"""
import os
import argparse
import time
import re
import h5py
import numpy as np
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement, islice
from ase.io import iread


# BP_DEFAULT = [6.0, 1.0, 1.0, 1.0, 1.0]


def initialize_argparser():
    """

    Initializes ArgumentParser for command-line operation.

    """
    argparser = argparse.ArgumentParser(description='Converts structures to \
                                        symmetry functions.')
    argparser.add_argument('action', choices=['represent', 'parse'])
    argparser.add_argument('input',
                           help='.hdf5 for representation or ase-compatible \
                                 file or root of filetree for parsing')
    argparser.add_argument('input2',
                           help='filename of property data to parse \
                           or parameters to use in fingerprint')
    argparser.add_argument('symbol_set',
                           help='comma-separated list of unique elements; \
                           e.g. "Au" or "Ba,Ti,O"')
    argparser.add_argument('-k', '--keyword',
                           help='keyword to parse embedded property data; \
                           default: None')
    argparser.add_argument('-o', '--output',
                           help='specify filename for .hdf5 output; \
                                default mirrors input name')
    argparser.add_argument('-i', '--index', default=':',
                           help='slicing using numpy convention; \
                           e.g. ":250" or "-500:" or "::2"')
    argparser.add_argument('-f', '--format',
                           help='file format for ase structure/molecule \
                           parsing. If unspecified, ASE will guess.')
    argparser.add_argument('-d', '--descriptor', choices=['BP', 'PH'],
                           default='PH',
                           help='method of representing data; \
                           default: placeholder (PH)')
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


def parse_property(prop_name, loose, keyword, index):
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
        prop_name (str): Property data filename.
        loose: True if crystal/molecule input is a directory.
        keyword (str): Optional string to find property values.
            (e.g "energy" to find "energy: 0.423eV"
            or "E0" to find "E0= -0.52E+02")
        index (str): Slice. Must match in parse_ase.

    Returns:
        property_list: Property values as floats or dictionary of
            filename-value pairs if loose.

    """
    with open(prop_name, 'r') as propfile:
        data = propfile.read()

    if loose:
        parser = re.compile('(\w+)(?:[=\s,:]+)(\S+)(?:[;,\s])')
        data_pairs = parser.findall(data)
        property_list = {k: v for k, v in data_pairs}
        # full dictionary, one value per structure file
    else:
        if keyword:
            parser = re.compile('(?:' + keyword
                                + '[=\s,:]*)([^;,\s]+)')

            property_list = [float(match) for match in parser.findall(data)]
        else:
            parser = re.compile('([^,;\s]+)(?:[,;\s])')
            property_list = [float(match) for match in parser.findall(data)]
            # list of values after slicing

            # assumed to match with ase.io.iread's slicing
    return property_list[strslice(index)]


def write_structure(h5f, structure, system_symbols, symbol_order, s_name,
                    property_value):
    """

    Writes one crystal/molecule object's data to .hdf5 file.

    Args:
        h5f: h5py object for writing.
        structure: ASE Atoms object.
        system_symbols: List of system-wide unique element names as strings.
        symbol_order: Dictionary of elements and specified order.
        s_name (str): Atoms' identifier to use as group name.
        property_value (float): Atoms' corresponding property value.

    """
    coords = structure.get_positions(wrap=False)
    natoms = len(coords)
    species = structure.get_chemical_symbols()
    symbol_set = list(set(species))
    assert set(species).issubset(set(system_symbols))
    species_counts = np.asarray([species.count(cspecie)
                                 for cspecie
                                 in symbol_set]).astype('i4')
    species, coords = zip(*sorted(zip(species, coords),
                                  key=lambda x: symbol_order.get(x[0])))
    try:
        unit = structure.get_cell(complete=False)
        periodic = True
    except (ValueError, RuntimeError, AttributeError):
        unit = np.zeros((3, 3))
        periodic = False
    dset_name = 'structures/%s/coordinates' % s_name
    dset_coords = h5f.create_dataset(dset_name,
                                     (natoms, 3),
                                     data=coords, dtype='f4',
                                     compression="gzip")
    dset_coords.attrs['natoms'] = natoms
    dset_coords.attrs['symbol_set'] = np.string_(symbol_set)
    dset_coords.attrs['species_counts'] = species_counts
    dset_coords.attrs['unit'] = unit
    dset_coords.attrs['periodic'] = periodic
    dset_coords.attrs['property'] = property_value


def parse_ase(input_name, output_name, index, system_symbols, form,
              property_list):
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
        system_symbols: List of system-wide unique element names as strings.
        form (str): Optional file format string to pass to ASE's read function.
        property_list: List of property values as floats or
            dictionary of filename-value pairs if loose.

    """
    prefix = input_name.replace('.', '') + '_'

    symbol_order = {k: v for v, k in enumerate(system_symbols)}
    with h5py.File(output_name, 'w', libver='latest') as h5f:
        try:
            s_count = len(h5f['structures'].keys())
        except KeyError:
            s_count = 0
        s_count_start = s_count + 0
        print(str(s_count_start), 'structures to start')
        loose = os.path.isdir(input_name)
        if loose:
            for root, dirs, files in slice_generator(os.walk(input_name),
                                                     index):
                for filename in files:
                    # print(os.path.join(root, filename))
                    try:
                        structures = iread(filename, format=form)
                    except (ValueError, IOError, IndexError):
                        continue
                    for structure in structures:
                        s_name = filename + str(s_count)
                        property_value = property_list[filename]
                        try:
                            write_structure(h5f, structure, system_symbols,
                                            symbol_order, s_name,
                                            property_value)
                            s_count += 1
                        except AssertionError:
                            continue
        else:
            structures = iread(input_name, format=form, index=index)
            for structure in structures:
                s_name = prefix + str(s_count)
                property_value = property_list[s_count]
                write_structure(h5f, structure, system_symbols,
                                symbol_order, s_name, property_value)
                s_count += 1
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
        property_value (float): Corresponding property value.

    """
    dset = h5i['structures'][s_name]['coordinates']
    coords = dset[()]
    symbol_set = [symbol.decode('utf-8')
                  for symbol in dset.attrs['symbol_set']]
    species_counts = dset.attrs['species_counts']

    species_list = []
    for symbol, ccount in zip(symbol_set, species_counts):
        species_list += [symbol] * ccount

    unit = dset.attrs['unit']
    periodic = dset.attrs['periodic']
    property_value = dset.attrs['property']

    return [coords, symbol_set, species_counts,
            species_list, unit, periodic, property_value]


def make_fingerprint(h5o, s_data, s_name, parameters,
                     system_symbols, descriptor):
    """

    Reads data for one crystal/molecule and corresponding property data
    from .hdf5 file.

    Args:
        h5o: h5py object for writing.
        s_data: List of data (output of read_structure).
        s_name (str): Atoms' identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.
        descriptor: Descriptor to use.

    """
    if descriptor == 'PH':
        placeholder_fingerprint(h5o, s_data, s_name, parameters,
                                system_symbols)
    elif descriptor == 'BP':
        bp_fingerprint(h5o, s_data, s_name, parameters, system_symbols)


def bp_fingerprint(h5o, s_data, s_name, parameters, system_symbols):
    """
    Parrinello-Behler representation. Computes fingerprints for
    pairwise (g_1) and triple (g_2) interactions.

    Fingerprints are ndarrays of size (n x s x k)
    where n = number of atoms, k = number of parameters given for the
    fingerprint, and s = combinations with replacement of the system's species
    set. When the input's species set is less than the system's species set,
    fingerprints are padded.

    Args:
        h5o: h5py object for writing.
        s_data: List of data (output of read_structure).
        s_name (str): Atoms' identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, symbol_set, species_counts,
     species_list, unit, periodic, property_value) = s_data
    assert set(symbol_set).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters

    bp = BehlerParrinello()
    bp._elements = symbol_set
    bp._element_pairs = list(combinations_with_replacement(symbol_set, 2))
    if periodic:
        bp._unitcell = unit

    distmatrix = cdist(coords, coords)
    costheta = bp.calculate_cosTheta(coords, distmatrix, periodic)

    g_1 = []
    for para in para_pairs:
        r_cut, r_s, eta, lambda_, zeta = para
        g_1.append(bp.g_1(distmatrix, species_list, periodic)[:, 0])

    g_2 = []
    for para in para_triplets:
        r_cut, r_s, eta, lambda_, zeta = para
        g_2.append(bp.g_2(costheta, distmatrix, species_list, periodic)[:, 0])

    g_1, g_2 = pad_fingerprints([g_1, g_2], symbol_set, system_symbols, [1, 2])

    dname_1 = 'structures/%s/pairs' % s_name
    dname_2 = 'structures/%s/triplets' % s_name
    h5o.create_dataset(dname_1, g_1.shape,
                       data=g_1, dtype='f4', compression="gzip")
    h5o.create_dataset(dname_2, g_2.shape,
                       data=g_2, dtype='f4', compression="gzip")
    h5o['structures'][s_name].attrs['natoms'] = len(coords)
    h5o['structures'][s_name].attrs['symbol_set'] = np.string_(symbol_set)
    h5o['structures'][s_name].attrs['species_counts'] = species_counts
    h5o['structures'][s_name].attrs['property'] = property_value


def placeholder_fingerprint(h5o, s_data, s_name, parameters, system_symbols):
    """

    Args:
        h5o: h5py object for writing.
        s_data: List of data (output of read_structure).
        s_name (str): Atoms' identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, symbol_set, species_counts,
     species_list, unit, periodic, property_value) = s_data
    assert set(symbol_set).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters
    n_atoms = len(coords)
    n_species = len(symbol_set)
    pair_num = len(list(combinations_with_replacement(range(n_species), 1)))
    triplet_num = len(list(combinations_with_replacement(range(n_species),
                                                         2)))

    # PLACEHOLDER CODE! Random g_1 and g_2, matching desired shape
    coord_sums = np.sum(coords, axis=1)
    # sum of coordinates for each atom

    pair_factors = np.random.rand(pair_num, 1)
    # 1 factor per pair interaction (specie in sphere)
    para_factors_1 = np.sum(para_pairs, axis=0)
    para_num_1 = len(para_factors_1)
    # 1 factor per parameter set
    tile_atoms_1 = np.tile(coord_sums.reshape(n_atoms, 1, 1),
                           (1, pair_num, para_num_1))
    tile_inter_1 = np.tile(pair_factors.reshape(1, pair_num, 1),
                           (n_atoms, 1, para_num_1))
    tile_parameters_1 = np.tile(para_factors_1.reshape(1, 1, para_num_1),
                                (n_atoms, pair_num, 1))
    g_1 = np.multiply(np.multiply(tile_atoms_1, tile_inter_1),
                      tile_parameters_1)
    # g_1.shape ~ (#atoms x #species x #pair_parameters)
    triplet_factors = np.random.rand(triplet_num, 1)
    # 1 factor per triplet interaction (2 species in sphere)
    para_factors_2 = np.sum(para_triplets, axis=0)
    para_num_2 = len(para_factors_2)
    # 1 factor per parameter set
    tile_atoms_2 = np.tile(coord_sums.reshape(n_atoms, 1, 1),
                           (1, triplet_num, para_num_2))
    tile_inter_2 = np.tile(triplet_factors.reshape(1, triplet_num, 1),
                           (n_atoms, 1, para_num_2))
    tile_parameters_2 = np.tile(para_factors_2.reshape(1, 1, para_num_2),
                                (n_atoms, triplet_num, 1))
    g_2 = np.multiply(np.multiply(tile_atoms_2, tile_inter_2),
                      tile_parameters_2)
    # g_2.shape ~ (#atoms x
    #              [#combinations of species with replacement] x
    #              #triplet_parameters

    # PLACEHOLDER CODE!
    g_1, g_2 = pad_fingerprints([g_1, g_2], symbol_set, system_symbols, [1, 2])

    dname_1 = 'structures/%s/pairs' % s_name
    dname_2 = 'structures/%s/triplets' % s_name
    h5o.create_dataset(dname_1, g_1.shape,
                       data=g_1, dtype='f4', compression="gzip")
    h5o.create_dataset(dname_2, g_2.shape,
                       data=g_2, dtype='f4', compression="gzip")
    h5o['structures'][s_name].attrs['natoms'] = len(coords)
    h5o['structures'][s_name].attrs['symbol_set'] = np.string_(symbol_set)
    h5o['structures'][s_name].attrs['species_counts'] = species_counts
    h5o['structures'][s_name].attrs['property'] = property_value


def pad_fingerprints(terms, symbol_set, system_symbols, dims):
    """

    Args:
        terms: List of fingerprints.
        symbol_set: List of unique element names in structure as strings.
        system_symbols: List of system-wide unique element names as strings.
        dims: Dimensionality of interaction.
            (e.g. 1 for pairwise, 2 for triplets)

    Returns:
        padded: Padded fingerprints.

    """
    assert len(dims) == len(terms)
    system_groups = [list(combinations_with_replacement(system_symbols, dim))
                     for dim in dims]

    s_groups = [list(combinations_with_replacement(symbol_set, dim))
                for dim in dims]

    n_groups = [len(groups) for groups in system_groups]

    padded = [np.zeros((g.shape[0], n_groups[j], g.shape[2]))
              for j, g in enumerate(terms)]

    content_slices = [[sys_group.index(group) for group in s_group]
                      for s_group, sys_group in zip(s_groups, system_groups)]

    for dim, content_indices in enumerate(content_slices):
        for s_index, sys_index in enumerate(content_indices):
            padded[dim][:, sys_index, :] = terms[dim][:, s_index, :]

    return padded

def apply_descriptors(input_name, para_file, output_name,
                      descriptor, system_symbols):
    """

    Reads crystal/molecule data from .hdf5, creates fingerprints, and writes to
    new .hdf5.

    Args:
        input_name (str): Filename of .hdf5 for reading crystal/molecule data.
        para_file (str): Filename of descriptor parameters.
        output_name (str): Filename of .hdf5 for writing fingerprint data.
        descriptor (str): Descriptor to use to represent crystal/molecule data.
        system_symbols: List of system-wide unique element names as strings.

    """
    # parameters = read_BP_parameters(para_file)
    kp = 4
    kt = 6
    parameters = (np.random.rand(5, kp), np.random.rand(5, kt))
    # Placeholder!

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
            h5o['system'].attrs = np.string_(system_symbols)

            s_names = list(h5i['structures'].keys())
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names):
                print('processing', str(j + 1).rjust(10), '/',
                      str(s_tot).rjust(10), end='\r')
                try:
                    s_data = read_structure(h5i, s_name)
                    make_fingerprint(h5o, s_data, s_name, parameters,
                                     system_symbols, descriptor)
                    s_count += 1
                except (KeyError, ValueError, AssertionError):
                    f_count += 1
                    continue
            print(str(s_count), 'fingerprints created')
            if f_count > 0:
                print(str(f_count), 'fingerprint(s) failed')


if __name__ == '__main__':
    argparser = initialize_argparser()
    args = argparser.parse_args()

    symbol_set = args.symbol_set.split(',')
    if not len(symbol_set) == len(set(list(symbol_set))):
        raise ValueError('duplicate elements in symbol_set: ' + symbol_set)

    if ':' not in args.index:
        raise ValueError('invalid slice specified: ' + args.index)

    input_name = args.input
    if args.output:
        output_name = args.output
        if '.hdf5' not in output_name:
            output_name = output_name + '.hdf5'
    else:
        output_name = input_name.split('.')[0] + '.hdf5'

    if args.action == 'represent':
        if '.hdf5' not in input_name:
            raise IOError('Expected .hdf5 input')

        t0 = time.time()
        apply_descriptors(input_name, args.input2, output_name,
                          args.descriptor, symbol_set)
        print(time.time() - t0)

    elif args.action == 'parse':
        loose = os.path.isdir(input_name)
        property_list = parse_property(args.input2, loose, args.keyword,
                                       index=args.index)
        # dictionary if loose, list otherwise
        parse_ase(input_name, output_name, args.index, symbol_set,
                  args.format, property_list)
