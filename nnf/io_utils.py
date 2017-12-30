"""

"""
import string
import numpy as np

ALPHABET = (string.ascii_uppercase + string.ascii_lowercase
            + string.digits)
ALPHABET_REVERSE = dict((c, i) for (i, c) in enumerate(ALPHABET))
BASE = len(ALPHABET)
SIGN_CHARACTER = '$'


def num_encode(n):
    if n < 0:
        return SIGN_CHARACTER + num_encode(-n)
    s = []
    while True:
        n, r = divmod(n, BASE)
        s.append(ALPHABET[r])
        if n == 0: break
    return ''.join(reversed(s))


def num_decode(s):
    if s[0] == SIGN_CHARACTER:
        return -num_decode(s[1:])
    n = 0
    for c in s:
        n = n * BASE + ALPHABET_REVERSE[c]
    return n


def generate_tag(unique_dict, add_on='0'):
    j_ints = ''.join([str(val) for val
                      in unique_dict.values()
                      if (isinstance(val, int)
                          and not (isinstance(val, bool)))])
    floats = [np.log(val) * 10
              for val in unique_dict.values()
              if (isinstance(val, float) and not val == 0)]

    float_strings = ['0{0:.0f}'.format(val) if val == abs(val)
                     else '{0:.0f}'.format(abs(val))
                     for val in floats]

    j_floats = ''.join(float_strings)
    tag = num_encode(int(j_ints + j_floats + add_on))
    return tag


def grid_string(entries, max_length=79, separator=', ', newline=';\n'):
    assert len(entries) > 0
    max_width = max([len(entry) for entry in entries])
    grid_lines = ''
    grid_line = str(entries.pop(0)).ljust(max_width)
    while len(entries) > 0:
        new_string = str(entries.pop(0)).ljust(max_width)
        if (len(grid_line) + len(new_string)) > max_length:
            grid_lines += grid_line + newline
            grid_line = new_string
        else:
            grid_line += separator + new_string
    return grid_lines


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
                   dset_types={}, **kwargs):
    """
    Writes datasets and attributes to specified group.

    Args:
        h5f: h5py file.
        group_path (str): Path to group
        dict_dsets (dict): Dataset names and ndarrays.
        dict_attrs (dict): Attribute names and values.
        dset_types (dict): Optional data types for dataset(s).
    """
    group = h5f.require_group(group_path)
    for dset_name, dset_data in dict_dsets.items():
        dtype = dset_types.get(dset_name, 'f4')
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
