"""
This module provides input/output functions for crystal/molecule data
(e.g. from VASP outputs).
"""
import os
import argparse
import re
import h5py
import traceback

import numpy as np
from itertools import islice
from ase.io import iread


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
    print(i)
    return islice(generator, *i)


def read_parameters(params_file):
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
    species, coords = zip(*sorted(zip(species, coords),
                                  key=lambda x: symbol_order.get(x[0])))
    element_set = sorted(list(set(species)),
                         key=symbol_order.get)
    assert set(species).issubset(set(sys_elements))
    species_counts = np.asarray([species.count(cspecie)
                                 for cspecie
                                 in sys_elements]).astype('i4')
    assert sum(species_counts) == natoms
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
    dset = h5f.require_group('structures/{}'.format(s_name))
    h5f['structures'][s_name].attrs['natoms'] = len(coords)
    h5f['structures'][s_name].attrs['element_set'] = np.string_(element_set)
    h5f['structures'][s_name].attrs['species_counts'] = species_counts
    h5f['structures'][s_name].attrs['unit'] = unit
    h5f['structures'][s_name].attrs['periodic'] = periodic
    h5f['structures'][s_name].attrs['energy'] = energy_val


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
                        s_name = str(s_count) + '_' + filename
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
                s_name = str(s_count) + '_' + input_name
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
