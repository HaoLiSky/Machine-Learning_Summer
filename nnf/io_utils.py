"""

"""
import h5py
import numpy as np


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
