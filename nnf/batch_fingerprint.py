"""
Input/output for fingerprints using descriptors.
"""
import os
import argparse
import traceback
import h5py
import numpy as np
from nnf.fingerprints import bp_fingerprint
from nnf.io_utils import slice_from_str, read_from_group, write_to_group
from nnf.batch_collate import CollatedMolecule
from nnf.framework import SettingsParser


class Fingerprint:
    """
    Reads fingerprints and attributes for one crystal/molecule
    and corresponding property data from .hdf5 file.

    Attributes:
       dsets_dict: Dictionary of fingerprint names and their ndarrays.
       natoms (int): Number of atoms in structure
       elements_set: List of unique element names in structure as strings.
       element_counts: List of occurences for each element type.
       elements_list: List of element namestring for each atom.
       energy_val (float): Corresponding property value.
    """

    def from_collated(self, mol, parameters, sys_elements, descriptor='bp',
                      derivs=False):
        """
        Reads data for one crystal/molecule and corresponding property data
        from .hdf5 file and writes fingerprint.

        Args:
            mol: CollatedMolecule object.
            parameters: List of descriptor parameters.
            sys_elements: List of system-wide unique element names as
                strings.
            descriptor (str): Descriptor to use.
                e.g. 'bp' for Behler-Parrinello
            derivs (boolean): Whether to calculate derivatives of
                fingerprints with respect to cartesian coordinates.
        """
        if descriptor == 'bp':
            inputs, shapes = bp_fingerprint(mol, parameters, sys_elements,
                                            derivs=derivs)
        else:
            raise ValueError

        self.dsets_dict = {label: term for label, term in inputs}
        self.attrs_dict = {'natoms'        : mol.natoms,
                           'element_set'   : np.string_(mol.element_set),
                           'element_counts': mol.element_counts,
                           'energy'        : mol.energy_val}
        self.natoms = mol.natoms
        self.elements_set = mol.element_set
        self.element_counts = mol.element_counts
        self.elements_list = []
        for symbol, ccount in zip(sys_elements, self.element_counts):
            self.elements_list += [symbol] * ccount

        self.fingerprint_shapes = np.string_('-'.join([str(shape)
                                                       for shape in shapes]))

    def from_file(self, h5f, path, sys_elements):
        """
        Args:
            h5f: h5py object for reading.
            path (str): path in h5f to molecule.
                e.g. 'fingerprints/Au55.1.2'
            sys_elements: List of system-wide unique element names as strings.
        """
        self.attrs_dict, self.dsets_dict = read_from_group(h5f, path)

        self.natoms = self.attrs_dict['natoms']
        self.elements_set = [symbol.decode('utf-8')
                             for symbol in self.attrs_dict['element_set']]
        self.element_counts = self.attrs_dict['element_counts']
        assert len(sys_elements) == len(self.element_counts)

        self.elements_list = []
        for symbol, ccount in zip(sys_elements, self.element_counts):
            self.elements_list += [symbol] * ccount
        assert len(self.elements_list) == self.natoms
        self.energy_val = self.attrs_dict['energy']

    def set_energy(self, energy_val):
        """
        Args:
            energy_val (float): Energy value.
        """
        self.attrs_dict['energy'] = energy_val
        self.energy_val = energy_val

    def to_file(self, h5o, path):
        """
        Save fingerprint to file.

        Args:
            h5o: h5py object for writing.
            path (str): Path to molecule subgroup
                e.g. 'fingerprints/Au55.1.2'
        """
        write_to_group(h5o, path, self.attrs_dict, self.dsets_dict)
        data = [np.string_(path.split('/')[-1]),
                np.string_(str(self.natoms)),
                np.string_('-'.join(self.elements_set)),
                np.string_('-'.join([str(x) for x in self.element_counts])),
                self.fingerprint_shapes,
                np.string_(self.energy_val)]

        return data


class FingerprintProcessor:
    """
    Top-level class to create fingerprints from collated
    crystal/molecule data. Reads from input .hdf5, creates fingerprints,
    and writes to output .hdf5.

    Args:
        output_name (str): Filename of .hdf5 for writing fingerprint data.
            e.g. fingerprints.h5
        parameters_file (str): Filename of descriptor parameter data.
            e.g. parameters.csv
        settings: Dictionary of settings.
    """

    def __init__(self, output_name, parameters_file, settings, **kwargs):
        self.output_name = output_name
        self.entries = []
        self.m_count = 0
        self.parameters = parameters_from_file(parameters_file)
        self.settings = settings
        self.settings.update(kwargs)
        self.sys_elements = self.settings['sys_elements']
        self.element_order = {k: v for v, k in enumerate(self.sys_elements)}

    def process_collated(self, filename):
        """
        Read collated molecules file. Creates fingerprint and writes to file
        for each collated molecule.

        Args:
            filename (str): Input file (e.g. collated.h5).
        """
        if os.path.isfile(self.output_name):
            self.combine_fingerprints(self.output_name, add_to_output=False)
        else:
            self.entries = []
            self.m_count = 0

        libver = self.settings['libver']
        index = self.settings['index']
        descriptor = self.settings['descriptor']
        derivs = self.settings['derivatives']
        with h5py.File(filename, 'r', libver=libver) as h5i:
            m_names = list(h5i.require_group('molecules').keys())
            with h5py.File(self.output_name, 'a', libver=libver) as h5o:
                m_tot = len(m_names)
                m_count_start = int(self.m_count)
                f_count = 0
                for j, m_name in enumerate(m_names[slice_from_str(index)]):
                    print('processing', str(j + 1).rjust(10), '/',
                          str(m_tot).rjust(10), end='\r')
                    try:
                        molpath = 'molecules/' + m_name
                        mol = CollatedMolecule()
                        mol.from_file(h5i, molpath, self.sys_elements)
                        fp = Fingerprint()
                        fp.from_collated(mol, self.parameters,
                                         self.sys_elements,
                                         descriptor=descriptor,
                                         derivs=derivs)
                        fp.set_energy(mol.energy_val)
                        m_name_new = '{}.{}'.format('.'.join(m_name.
                                                             split('.')[:-1]),
                                                    self.m_count)
                        path = 'fingerprints/' + m_name_new
                        entry = fp.to_file(h5o, path)
                        self.entries.append(entry)
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
                self.update_system_details(h5o)

    def combine_fingerprints(self, filename, add_to_output=True):
        """
        Load a fingerprints file and copy all fingerprint data to output file.
        If processor's output and fingerprints file are the same, only
        load the element list and entries list.

        Args:
            filename (str): Fingerprints file (e.g. fingerprints.h5)
            add_to_output (bool): Whether to copy all groups from file or
                simply load entries and system elements.
        """
        libver = self.settings['libver']
        with h5py.File(filename, 'a', libver=libver) as h5i:
            m_count = len(h5i.require_group('fingerprints').keys())
            m_count_start = int(m_count)
            print(m_count_start, 'fingerprints in file.')
            try:
                file_sys_elements = [symbol.decode('utf-8')
                                     for symbol
                                     in h5i['system'].attrs['sys_elements']]
                if not self.sys_elements:
                    self.sys_elements = file_sys_elements
                else:
                    assert self.sys_elements == file_sys_elements
                entries_list = h5i['system/fingerprint_entries'][()].tolist()
                self.entries.extend([line.split(b';')
                                     for line in entries_list])
            except KeyError:
                pass
            if add_to_output and not self.output_name == filename:
                with h5py.File(self.output_name, libver=libver) as h5o:
                    m_names = h5i.require_group('molecules').keys()
                    for j, m_name in enumerate(m_names):
                        group_path = 'molecules/' + m_name
                        src = h5i.require_group(group_path)
                        dest = h5o.require_group(group_path)
                        src.copy(dest)
                self.update_system_details(h5i)

    def update_system_details(self, h5f):
        """
        Updates system element list, descriptor parameters, and entries
        in hdf5.

        Args:
            h5f: h5py object.
        """
        entries = np.asarray([b';'.join(entry)
                              for entry in self.entries])
        write_to_group(h5f, 'system',
                       {'sys_elements': np.string_(self.sys_elements)},
                       {'pair_params'   : self.parameters[0],
                        'triplet_params': self.parameters[1]})

        write_to_group(h5f, 'system',
                       {},
                       {'fingerprint_entries': entries},
                       dset_types={'fingerprint_entries': entries.dtype},
                       maxshape=(None,))
        self.summary()

    def summary(self):
        """
        Print fingerprint details.
        """
        sizes = [int(entry[1]) for entry in self.entries]
        compositions = np.asarray([entry[3].decode('utf-8')
                                   for entry in self.entries])
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
        """
        Export fingerprint entries to comma-separated values file.

        Args:
            filename (str): Output file (e.g. fingerprints.csv)
        """
        header = 'Name,Size,Elements,Composition,Fingerprint Shapes,Energy;\n'
        with open(filename, 'w') as fil:
            lines = [b','.join(line).decode('utf-8')
                     for line in self.entries]
            text = header + ';\n'.join(lines)
            fil.write(text)


def parameters_from_file(params_file):
    """
    Read parameter sets from file.

    Args:
        params_file (str): Parameters filename. (e.g. parameters.csv)

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


if __name__ == '__main__':
    description = 'Create fingerprints from collated molecules file.'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('--settings_file', '-s', default='settings.cfg',
                           help='Filename of settings.')
    argparser.add_argument('--verbosity', '-v', action='count')
    argparser.add_argument('--export', '-E', action='store_true',
                           help='Export entries to csv.')
    args = argparser.parse_args()
    settings = SettingsParser('Fingerprint').read(args.settings_file)
    settings['verbosity'] = args.verbosity
    input_name = settings['inputs_name']
    output_name = settings['outputs_name']
    parameters_file = settings['parameters_file']
    sys_elements = settings['sys_elements']
    assert sys_elements != ['None']

    processor = FingerprintProcessor(output_name, parameters_file,
                                     settings)
    processor.process_collated(input_name)
    if args.export:
        processor.export_entries(input_name.split('.')[0] + '.csv')
