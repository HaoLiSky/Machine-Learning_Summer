"""

"""
import h5py
import numpy as np
from itertools import islice


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


def read_from_group(h5f, group_path):
    """

    Read all datasets and attributes from specified group.

    Args:
        h5f: h5py file.
        group_path (str): Path to group.

    Returns:
        dict_dsets (dict): Dataset names and ndarrays.
        dict_attrs (dict): Attribute names and values.

    """
    group = h5f[group_path]
    dset_names = list(group.keys())
    attr_names = list(group.attrs.keys())

    dict_dsets = {dset: group[dset][()] for dset in dset_names}

    dict_attrs = {attr: group.attrs[attr] for attr in attr_names}

    return dict_attrs, dict_dsets


def write_to_group(h5f, group_path, dict_attrs, dict_dsets,
                   dict_dset_types={}, **kwargs):
    """

    Writes datasets and attributes to specified group.

    Args:
        h5f: h5py file.
        group_path (str): Path to group
        dict_dsets (dict): Dataset names and ndarrays.
        dict_attrs (dict): Attribute names and values.
        dict_dset_types (dict): Optional data types for dataset(s).

    """
    group = h5f.require_group(group_path)
    for dset_name, dset_data in dict_dsets.items():
        dtype = dict_dset_types.get(dset_name, 'f4')
        try:
            dset = group[dset_name]
            dset.resize(len(dset_data), axis=0)
        except KeyError:
            dset = group.require_dataset(dset_name, dset_data.shape,
                                         dtype=dtype, compression='gzip',
                                         **kwargs)
        dset[...] = dset_data

    for attr_name, attr_data in dict_attrs.items():
        group.attrs[attr_name] = attr_data


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
    default_dict = {'natoms'        : 0,
                    'energy'        : '',
                    'element_set'   : [],
                    'element_counts': [0],
                    'unit_vectors'  : [],
                    'periodic'      : False}

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
                                    [dset[fp][()].shape[-1] for fp in
                                     dset.keys()],
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
