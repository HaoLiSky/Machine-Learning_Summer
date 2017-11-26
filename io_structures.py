"""
This module provides input/output functions for crystal/molecule data
(e.g. from VASP outputs).
"""
import os
import re
import h5py
import traceback

import numpy as np
from itertools import islice
from ase.io import iread
from io_hdf5 import write_to_group, read_from_group


def collate_structures(input_name, output_name, sys_elements,
                       energy_data, index=':', form=None):
    """

    Top-level function to parse crystal/molecule data and property data
    and write to .hdf5 file.

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

    element_order = {k: v for v, k in enumerate(sys_elements)}
    with h5py.File(output_name, 'a', libver='latest') as h5f:
        # initial number of groups in h5f
        s_count = len(h5f.require_group('structures').keys())
        s_count_start = s_count + 0
        print(s_count_start, 'structures in file to start.')
        try:
            sys_entries = h5f['system/sys_entries'][()].tolist()
            sys_entries = [line.split(b';') for line in sys_entries]
        except KeyError:
            sys_entries = []

        loose = os.path.isdir(input_name)
        if loose:
            for root, dirs, files in os.walk(input_name):
                sorted_files = sorted(files)
                for filename in sorted_files[slice_from_str(index)]:
                    try:
                        structures = iread(os.path.join(root, filename),
                                           format=form)
                    except (ValueError, IOError, IndexError):
                        continue
                    for structure in structures:
                        s_name = '{}_{}'.format(filename, s_count)
                        try:
                            s_energy_val = energy_data[filename]
                            s_data = parse_structure(h5f, structure,
                                                     sys_elements,
                                                     element_order, s_name,
                                                     s_energy_val)
                            sys_entries.append(s_data)
                            s_count += 1
                            print('read', s_count - s_count_start, end='\r')
                        except KeyError:
                            print('no corresponding energy_data:', s_name)
                            continue
                        except AssertionError:
                            print('error in', s_name, end='\n\n')
                            traceback.print_exc()
                            continue
        else:  # not loose files
            structures = iread(input_name, format=form, index=index)
            for structure in structures:
                s_name = '{}_{}'.format(input_name, s_count)
                try:
                    s_energy_val = energy_data[s_count - s_count_start]
                    s_data = parse_structure(h5f, structure, sys_elements,
                                             element_order, s_name,
                                             s_energy_val)
                    sys_entries.append(s_data)
                    s_count += 1
                except IndexError:
                    print('no corresponding energy_data:', s_name)
                    continue
                except AssertionError:
                    print('error in', s_name, end='\n\n')
                    traceback.print_exc()
                    continue
                print('read', s_count - s_count_start, end='\r')
        print(s_count - s_count_start, 'new structures parsed into',
              output_name)

        sys_entries = np.asarray(sys_entries)
        energy_indices = np.asarray([float(x)
                                     for x in sys_entries[:, -1]])
        sys_entries = sys_entries[np.lexsort((energy_indices,
                                              sys_entries[:, 3]))]
        s_names_list = sys_entries[:, 0]
        sys_entries = np.asarray([b';'.join(entries)
                                  for entries in sys_entries])
        write_to_group(h5f, 'system',
                       {'sys_elements': np.string_(sys_elements)},
                       {})

        write_to_group(h5f, 'system',
                       {},
                       {'sys_entries': sys_entries,
                        's_names_list': s_names_list},
                       dict_dset_types={'s_names_list': s_names_list.dtype,
                                        'sys_entries': sys_entries.dtype},
                       maxshape=(None,))


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
        energy_list = {k: v.replace(';', '')
                       for k, v in data_pairs}
        # full dictionary, one value per structure file
        return energy_list
    else:
        if keyword:
            parser = re.compile('(?:' + keyword
                                + '[=\s,:]*)([^;,\s]+)')
            energy_list = [match.replace(';', '')
                           for match in parser.findall(data)]

        else:
            parser = re.compile('([^,;\s]+)(?:[,;\s])')
            energy_list = [match.replace(';', '')
                           for match in parser.findall(data)]
            # list of values after slicing
            # assumed to match with ase.io.iread's slicing
        return energy_list[slice_from_str(index)]


def parse_structure(h5f, structure, sys_elements, element_order, s_name,
                    energy_val):
    """

    Writes one crystal/molecule object's data to .hdf5 file.

    Args:
        h5f: h5py object for writing.
        structure: ASE Atoms object.
        sys_elements: List of system-wide unique element names as strings.
        element_order: Dictionary of elements and specified order.
        s_name (str): Atoms' identifier to use as group name.
        energy_val (float): Atoms' corresponding property value.

    """
    s_coords = structure.get_positions(wrap=False)
    natoms = len(s_coords)
    s_elements = structure.get_chemical_symbols()
    s_elements, coords = zip(*sorted(zip(s_elements, s_coords),
                                     key=lambda x: element_order.get(x[0])))
    s_element_set = sorted(list(set(s_elements)),
                           key=element_order.get)
    assert set(s_element_set).issubset(set(sys_elements))
    s_element_counts = np.asarray([s_elements.count(element)
                                   for element
                                   in sys_elements]).astype('i4')
    assert sum(s_element_counts) == natoms
    try:
        s_unit_vectors = structure.get_cell(complete=False)
        s_periodic = True
    except (ValueError, RuntimeError, AttributeError):
        s_unit_vectors = np.zeros((3, 3))
        s_periodic = False
    if np.all(np.isclose(s_unit_vectors,
                         np.zeros((3, 3)))):
        s_periodic = False

    dict_dsets = {'coordinates': s_coords}
    dict_attrs = {'natoms': natoms,
                  'element_set': np.string_(s_element_set),
                  'element_counts': s_element_counts,
                  'unit_vectors': s_unit_vectors,
                  'periodic': s_periodic,
                  'energy': energy_val}
    group_name = 'structures/{}'.format(s_name)
    write_to_group(h5f, group_name, dict_attrs, dict_dsets)
    s_data = [np.string_(s_name),
              np.string_(str(natoms)),
              np.string_(','.join(s_element_set)),
              np.string_(','.join([str(x) for x in s_element_counts])),
              np.string_(str(s_periodic)),
              np.string_(str(s_coords.shape)),
              np.string_(energy_val)]

    return s_data


def read_collated_structure(h5f, s_name, sys_elements):
    """

    Reads data for one crystal/molecule and correpsonding property data
    from .hdf5 file.

    Args:
        h5f: h5py object for reading.
        s_name (str): Group name in h5f corresponding to Atoms' identifier.
        sys_elements: List of system-wide unique element names as strings.

    Returns:
        coords (n x 3 numpy array): Coordinates for each atom.
        element_set: List of unique element names in structure as strings.
        elements_counts: List of occurences for each element type.
        elements_list: List of element namestring for each atom.
        unit (3x3 numpy array): Atoms' unit cell vectors (zeros if molecule).
        periodic: True if crystal.
        energy_val (float): Corresponding property value.

    """
    attrs_dict, dsets_dict = read_from_group(h5f,
                                             'structures/{}'.format(s_name))

    coords = dsets_dict['coordinates']
    element_set = [symbol.decode('utf-8')
                   for symbol in attrs_dict['element_set']]
    element_counts = attrs_dict['element_counts']
    assert len(sys_elements) == len(element_counts)
    elements_list = []
    for symbol, ccount in zip(sys_elements, element_counts):
        elements_list += [symbol] * ccount
    assert len(elements_list) == len(coords)
    unit_vectors = attrs_dict['unit_vectors']
    periodic = attrs_dict['periodic']
    energy_val = attrs_dict['energy']

    return [coords, element_set, element_counts,
            elements_list, unit_vectors, periodic, energy_val]


def slice_from_str(string):
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
