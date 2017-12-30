"""
This module provides input/output functions for fingerprints
using descriptors.
"""
import os
import h5py
import traceback
import numpy as np
from nnf.fingerprints import bp_fingerprint
from nnf.io_utils import slice_from_str, read_from_group, write_to_group
from nnf.io_structures import CollatedMolecule


class Fingerprinter:
    """
    Top-level function to create fingerprints from collated
    crystal/molecule
    data. Reads from input .hdf5, creates fingerprints, and writes to
    output
     .hdf5.

    Args:
        output_name (str): Filename of .hdf5 for writing fingerprint data.
        parameters_file (str): Filename of descriptor parameter data.
        sys_elements: List of system-wide unique element names as strings.
    """
    def __init__(self, output_name, parameters_file, sys_elements):
        self.output_name = output_name
        self.sys_elements = sys_elements
        self.element_order = {k: v for v, k in enumerate(sys_elements)}
        self.entries = []
        self.m_count = 0
        self.parameters = read_parameters_from_file(parameters_file)

    def make_fingerprints(self, filename,
                          descriptor, index, derivs=False):
        """
        filename (str): Filename of .hdf5 for reading crystal/molecule
        data.
        descriptor (str): Descriptor to use to represent
        crystal/molecule data.
        index (str): Slice. Defaults to ':' for all entries.
        derivs (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.
        """
        if os.path.isfile(self.output_name):
            self.combine_fingerprints(self.output_name, add_to_output=False)
        else:
            self.entries = []
            self.m_count = 0
        with h5py.File(filename, 'r', libver='latest') as h5i:
            m_names = h5i.require_group('molecules').keys()
            with h5py.File(self.output_name, 'a', libver='latest') as h5o:
                m_tot = len(m_names)
                m_count_start = int(self.m_count)
                f_count = 0
                for j, m_name in enumerate(m_names[slice_from_str(index)]):
                    print('processing', str(j + 1).rjust(10), '/',
                          str(m_tot).rjust(10), end='\r')
                    try:
                        path = 'molecules/' + m_name
                        mol = CollatedMolecule.from_file(h5i,
                                                         path,
                                                         self.sys_elements)
                        m_name_new = '{}.{}'.format('_'.join(m_name.
                                                             split('_')[:-1]),
                                                    self.m_count)
                        f_data = make_fingerprint(h5o, mol,
                                                  m_name_new,
                                                  self.parameters,
                                                  self.sys_elements,
                                                  descriptor=descriptor,
                                                  derivs=derivs)
                        self.entries.append(f_data)
                        self.m_count += 1
                    except AssertionError:
                        print('error in', m_name, end='\n\n')
                        traceback.print_exc()
                        f_count += 1
                        continue
                print(str(m_count_start - self.m_count),
                      'new fingerprints created in', self.output_name)
                if f_count > 0:
                    print(str(f_count), 'fingerprint(s) failed')
            self.update_system_details(h5i)

    def combine_fingerprints(self, filename, add_to_output=True):
        with h5py.File(filename, 'a', libver='latest') as h5i:
            m_count = len(h5i.require_group('fingerprints').keys())
            m_count_start = int(m_count)
            print(m_count_start, 'fingerprints in file.')
            try:
                entries_list = h5i['system/fingerprint_entries'][()].tolist()
                self.entries.extend([line.split(b';')
                                     for line in entries_list])
            except KeyError:
                pass
            if add_to_output:
                with h5py.File(self.output_name, libver='latest') as h5o:
                    m_names = h5i.require_group('molecules').keys()
                    for j, m_name in enumerate(m_names):
                        group_path = 'molecules/' + m_name
                        src = h5i.require_group(group_path)
                        dest = h5o.require_group(group_path)
                        src.copy(dest)
                self.update_system_details(h5i)

    def update_system_details(self, h5f):
        entries = np.asarray([b';'.join(entry)
                              for entry in self.entries])
        write_to_group(h5f, 'system',
                       {'sys_elements': np.string_(self.sys_elements)},
                       {'pair_params'   : self.parameters[0],
                        'triplet_params': self.parameters[1]})

        write_to_group(h5f, 'system',
                       {},
                       {'fingerprint_entries': entries},
                       dict_dset_types={'fingerprint_entries':
                                            entries.dtype},
                       maxshape=(None,))
        self.summary()

    def summary(self):
        sizes = [int(entry[1]) for entry in self.entries]
        compositions = [entry[3].decode('utf-8') for entry in self.entries]
        print('Entries:', len(self.entries))
        shape_set = set([entry[4] for entry in self.entries])
        shapes = [shape.decode('utf-8') for shape in shape_set]
        energies = [float(entry[-1]) for entry in self.entries]
        print('System Elements:', self.sys_elements)
        print('Min Size:', np.min(sizes), ':', compositions[np.argmin(sizes)])
        print('Max Size:', np.max(sizes), ':', compositions[np.argmax(sizes)])
        print('Size std:', np.std(sizes))
        print('Fingerprint Shapes:', ';'.join(shapes))
        print('Min energy:', np.min(energies))
        print('Max energy:', np.max(energies))
        print('Energy std:', np.std(energies))

    def export_entries(self, filename):
        header = 'Name,Size,Elements,Composition,Fingerprint Shapes,Energy;\n'
        with open(filename, 'w') as fil:
            lines = [b','.join(line).decode('utf-8')
                     for line in self.entries]
            text = header + ';\n'.join(lines)
            fil.write(text)


def make_fingerprint(h5f, mol, m_name, parameters,
                     sys_elements, descriptor='bp',
                     derivs=False):
    """
    Reads data for one crystal/molecule and corresponding property data
    from .hdf5 file and writes fingerprint.

    Args:
        h5f: h5py object for writing.
        mol: CollatedMolecule object.
        m_name (str): Molecule's identifier to be used as group name in h5o.
        parameters: Descriptor parameters.
        sys_elements: List of system-wide unique element names as strings.
        descriptor: Descriptor to use.
        derivs (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.
    """

    if descriptor == 'bp':
        inputs, shapes = bp_fingerprint(mol, parameters, sys_elements,
                                        derivs=derivs)
    else:
        raise ValueError

    dict_dsets = {label: term for label, term in inputs}
    dict_attrs = {'natoms'        : len(mol.coords),
                  'element_set'   : np.string_(mol.element_set),
                  'element_counts': mol.element_counts,
                  'energy'        : mol.energy_val}
    group_name = 'fingerprints/' + m_name
    write_to_group(h5f, group_name, dict_attrs, dict_dsets)

    fingerprint_shapes = np.string_(','.join([str(shape)
                                              for shape in shapes]))
    f_data = [np.string_(mol.m_name),
              np.string_(str(len(mol.coords))),
              np.string_(','.join(mol.element_set)),
              np.string_(','.join([str(x) for x in mol.element_counts])),
              fingerprint_shapes,
              np.string_(mol.energy_val)]

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


class Fingerprint:
    """
    Reads fingerprints and attributes for one crystal/molecule
    and correpsonding property data from .hdf5 file.
    
    Attributes:
       dsets_dict: Dictionary of fingerprint names and their ndarrays.
       natoms (int): Number of atoms in structure
       elements_set: List of unique element names in structure as strings.
       element_counts: List of occurences for each element type.
       elements_list: List of element namestring for each atom.
       energy_val (float): Corresponding property value.
    """
    def __init__(self):
        self.attrs_dict = {}
        self.dsets_dict = {}
        self.natoms = 0
        self.elements_set = []
        self.element_counts = []
        self.elements_list = []
        self.energy_val = 0.0
        
    def from_file(self, h5f, path, sys_elements):
        """
        Args:
            h5f: h5py object for reading.
            path (str): path in h5f to molecule.
            sys_elements: List of system-wide unique element names as strings.
        """
        self.attrs_dict, self.dsets_dict = read_from_group(h5f, path)

        self.natoms = self.attrs_dict['natoms']
        self.elements_set = [symbol.decode('utf-8')
                             for symbol in self.attrs_dict['element_set']]
        self.element_counts = self.attrs_dict['element_counts']
        assert len(sys_elements) == len(self.element_counts)

        for symbol, ccount in zip(sys_elements, self.element_counts):
            self.elements_list += [symbol] * ccount
        assert len(self.elements_list) == self.natoms
        self.energy_val = self.attrs_dict['energy']

    def to_file(self, h5o, path):
        write_to_group(h5o, path, self.attrs_dict, self.dsets_dict)
