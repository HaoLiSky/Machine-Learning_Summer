"""
Input/output for crystal/molecule data (e.g. from VASP outputs).
"""
import os
import re
import argparse
import traceback
import h5py
import numpy as np
from ase.io import iread
from nnf.io_utils import slice_from_str, write_to_group, read_from_group
from nnf.framework import SettingsParser


class CollatedMolecule:
    """
    Attributes:
        coords (n x 3 numpy array): Coordinates for each atom.
        element_set: List of unique element names in structure as strings.
        element_counts: List of occurences for each element type.
        elements_list: List of element namestring for each atom.
        unit_vectors (3x3 numpy array): Atoms' unit cell vectors (zeros if
        molecule).
        periodic: True if crystal.
        energy_val (float): Corresponding property value.
    """

    def from_ase(self, atoms, sys_elements, element_order):
        """
        Reads crystal/molecule data from an ase Atoms object.

        Args:
            atoms: ASE Atoms object.
            sys_elements: List of system-wide unique element names as strings.
            element_order: Dictionary of elements and specified order.
        """
        self.coords = atoms.get_positions(wrap=False)
        self.natoms = len(self.coords)
        elements = atoms.get_chemical_symbols()
        elements, coords = zip(*sorted(zip(elements, self.coords),
                                       key=lambda x: element_order.get(x[0])))
        # ensure that molecule belongs in system
        self.element_set = sorted(list(set(elements)),
                                  key=element_order.get)
        assert set(self.element_set).issubset(set(sys_elements))
        # ensure that coordinate array size matches composition
        self.element_counts = np.asarray([elements.count(element)
                                          for element
                                          in sys_elements]).astype('i4')
        assert sum(self.element_counts) == self.natoms
        self.elements_list = []
        for symbol, ccount in zip(sys_elements, self.element_counts):
            self.elements_list += [symbol] * ccount
        assert len(self.elements_list) == len(self.coords)

        try:
            self.unit_vectors = atoms.get_cell(complete=False)
            self.periodic = True
        except (ValueError, RuntimeError, AttributeError):
            self.unit_vectors = np.zeros((3, 3))
            self.periodic = False

        if np.all(np.isclose(self.unit_vectors, np.zeros((3, 3)))):
            self.periodic = False

        self.attrs_dict = {'natoms'        : self.natoms,
                           'element_set'   : np.string_(self.element_set),
                           'element_counts': self.element_counts,
                           'unit_vectors'  : self.unit_vectors,
                           'periodic'      : self.periodic}
        self.dsets_dict = {'coordinates': np.asarray(coords)}

    def from_file(self, h5f, path, sys_elements):
        """
        Args:
            h5f: h5py object for reading.
            path (str): Path in h5f to molecule.
                e.g. 'molecules/au55.1.2'
            sys_elements: List of system-wide unique element names as strings.
        """
        self.attrs_dict, self.dsets_dict = read_from_group(h5f, path)

        self.coords = self.dsets_dict['coordinates']
        self.natoms = len(self.coords)
        self.element_set = [symbol.decode('utf-8')
                            for symbol in self.attrs_dict['element_set']]
        self.element_counts = self.attrs_dict['element_counts']
        assert len(sys_elements) == len(self.element_counts)
        self.elements_list = []
        for symbol, ccount in zip(sys_elements, self.element_counts):
            self.elements_list += [symbol] * ccount
        assert len(self.elements_list) == len(self.coords)
        self.unit_vectors = self.attrs_dict['unit_vectors']
        self.periodic = self.attrs_dict['periodic']
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
        Save collated molecule to file.

        Args:
            h5o: h5py object for writing.
            path (str): Path to molecule subgroup.
                e.g. 'molecules/au55.1.2'
        """
        write_to_group(h5o, path, self.attrs_dict, self.dsets_dict)
        data = [np.string_(path.split('/')[-1]),
                np.string_(str(np.sum(self.element_counts))),
                np.string_('-'.join(self.elements_list)),
                np.string_('-'.join([str(x) for x in self.element_counts])),
                np.string_(str(self.periodic)),
                np.string_(str(self.coords.shape).replace(',', '-')),
                np.string_(self.energy_val)]
        return data


class BatchCollator:
    """
    Top-level function to parse crystal/molecule data and property data
    and write to .hdf5 file.

    Data is stored in one group per crystal/molecule, named by the parent
    filename and the id of the crystal/molecule, assigned in order of
    writing.

    Example: the 31st dataset in the h5py object "h5f", corresponding to a
        molecule parsed from "au55.xyz", is stored in the .hdf5 location
        "h5f/structures/au55.xyz.31/coordinates."

    Args:
        output_name (str): Filename for .hdf5.
        settings: Dictionary of settings.
        sys_elements: List of system-wide unique element names as strings.
    """

    def __init__(self, output_name, settings, sys_elements, **kwargs):
        self.output_name = output_name
        self.sys_elements = sys_elements
        self.element_order = {k: v for v, k in enumerate(sys_elements)}
        self.entries = []
        self.sources = []
        self.m_count = 0
        self.settings = settings
        self.settings.update(kwargs)

    def parse_molecules(self, filename, energy_file):
        """
        Args:
            filename (str): Filename of crystal/molecule data.
            energy_file: File with property values as floats.
        """
        libver = self.settings['libver']
        index = self.settings['index']
        keyword = self.settings['keyword']
        form = self.settings['input_format']
        if form == '':
            form = None
        if os.path.isfile(self.output_name):
            self.combine_collated(self.output_name, add_to_output=False)
        else:
            self.entries = []
            self.sources = []
            self.m_count = 0
        m_count_start = int(self.m_count)
        with open(energy_file, 'r') as energy_handle:
            energy_text = energy_handle.read()
        energy_data = find_energy_in_text(energy_text, keyword=keyword)
        with h5py.File(self.output_name, 'a', libver=libver) as h5f:
            molecules = iread(filename, format=form, index=index)
            for molecule in molecules:
                m_name = '{}.{}'.format(filename, self.m_count)
                try:
                    path = 'molecules/{}'.format(m_name)
                    energy_val = energy_data[self.m_count - m_count_start]
                    mol = CollatedMolecule()
                    mol.from_ase(molecule, self.sys_elements,
                                 self.element_order)
                    mol.set_energy(energy_val)
                    entry = mol.to_file(h5f, path)
                    self.entries.append(entry)
                    self.m_count += 1
                    print('read', self.m_count - m_count_start, end='\r')
                except IndexError:
                    print('no corresponding energy_data:', m_name)
                    continue
                except AssertionError:
                    print('error in', m_name, end='\n\n')
                    traceback.print_exc()
                    continue
            self.sources.append(filename)
            self.update_system_details(h5f)

    def parse_loose_molecules(self, directory, energy_file):
        """
        Args:
            directory (str): Root directory of filetree of
                crystal/molecule data.
            energy_file: Dictionary of filename-value pairs.
        """
        libver = self.settings['libver']
        index = self.settings['index']
        form = self.settings['input_format']
        if form == '':
            form = None
        if os.path.isfile(self.output_name):
            self.combine_collated(self.output_name, add_to_output=False)
        else:
            self.entries = []
            self.sources = []
            self.m_count = 0
        m_count_start = int(self.m_count)
        with open(energy_file, 'r') as energy_handle:
            energy_text = energy_handle.read()
        energy_data = find_energy_in_text(energy_text, loose=True)
        with h5py.File(self.output_name, 'a', libver=libver) as h5f:
            for root, dirs, files in os.walk(directory):
                sorted_files = sorted(files)
                for filename in sorted_files[slice_from_str(index)]:
                    try:
                        molecules = iread(os.path.join(root, filename),
                                          format=form)
                    except (ValueError, IOError, IndexError):
                        continue
                    for molecule in molecules:
                        m_name = '{}.{}'.format(filename, self.m_count)
                        try:
                            path = 'molecules/{}'.format(m_name)
                            energy_val = energy_data[filename]
                            mol = CollatedMolecule()
                            mol.from_ase(molecule,  self.sys_elements,
                                         self.element_order)
                            mol.set_energy(energy_val)
                            entry = mol.to_file(h5f, path)
                            self.entries.append(entry)
                            self.sources.append(filename)
                            self.m_count += 1
                            print('read', self.m_count - m_count_start,
                                  end='\r')
                        except KeyError:
                            print('no corresponding energy_data:', m_name)
                            continue
                        except AssertionError:
                            print('error in', m_name, end='\n\n')
                            traceback.print_exc()
                            continue
            self.update_system_details(h5f)

    def combine_collated(self, filename, add_to_output=True):
        """
        Load a fingerprints file and copy all data to output file.
        If BatchCollator's output and fingerprints file are the same, only
        load the element list and entries list.

        Args:
            filename (str): Collated molecules file.
            add_to_output (bool): Whether to copy all groups from file or
                simply load entries and system elements.
        """
        libver = self.settings['libver']
        with h5py.File(filename, 'a', libver=libver) as h5i:
            m_count = len(h5i.require_group('molecules').keys())
            m_count_start = int(m_count)
            print(m_count_start, 'collated molecules in file.')
            try:
                file_sys_elements = [symbol.decode('utf-8')
                                     for symbol
                                     in h5i['system'].attrs['sys_elements']]
                assert self.sys_elements == file_sys_elements
                entries_list = h5i['system/collated_entries'][()].tolist()
                self.entries.extend([line.split(b';')
                                     for line in entries_list])
                sources_list = h5i['system/collated_sources'][()].tolist()
                self.sources.extend([line.split(b';')
                                     for line in sources_list])
            except KeyError:
                pass
            if add_to_output:
                with h5py.File(self.output_name, libver=libver) as h5o:
                    m_names = h5i.require_group('molecules').keys()
                    for j, m_name in enumerate(m_names):
                        group_path = 'molecules/' + m_name
                        src = h5i.require_group(group_path)
                        dest = h5o.require_group(group_path)
                        src.copy(dest)
                        m_count += 1
                print(m_count - m_count_start, ' added.')
                self.m_count += m_count
                self.update_system_details(h5i)

    def update_system_details(self, h5f):
        """
        Updates system element list, sources list, and entries
        in hdf5.

        Args:
            h5f: h5py object.
        """
        entries = np.asarray(self.entries)

        energy_indices = np.asarray([float(x)
                                     for x in entries[:, -1]])
        entries = entries[np.lexsort((energy_indices,
                                      entries[:, 3]))]
        entries = np.asarray([b';'.join(line)
                              for line in entries])
        sources = np.string_(self.sources)
        write_to_group(h5f, 'system',
                       {'sys_elements': np.string_(self.sys_elements)},
                       {'collated_entries': entries,
                        'collated_sources': sources},
                       dict_dset_types={'collated_entries': entries.dtype,
                                        'collated_sources': sources.dtype},
                       maxshape=(None,))
        self.summary()

    def summary(self):
        """
        Print collated molecules details.
        """
        print('Entries:', len(self.entries))
        sizes = [int(entry[1]) for entry in self.entries]
        compositions = np.asarray([entry[3].decode('utf-8')
                                   for entry in self.entries])
        energies = [float(entry[-1]) for entry in self.entries]
        print('System Elements:', self.sys_elements)
        print('Min Size:', np.min(sizes), ':', compositions[np.argmin(sizes)])
        print('Max Size:', np.max(sizes), ':', compositions[np.argmax(sizes)])
        print('Size std:', np.std(sizes))
        print('Unique Compositions:', len(set(compositions)))
        print('Min energy:', np.min(energies))
        print('Max energy:', np.max(energies))
        print('Energy std:', np.std(energies))

    def export_entries(self, filename):
        """
        Export entries to comma-separated values file.

        Args:
            filename (str): Filename for .csv
        """
        header = 'Name,Size,Elements,Composition,Coords Shape,Energy;\n'
        with open(filename, 'w') as fil:
            lines = [b','.join(line).decode('utf-8')
                     for line in self.entries]
            text = header + ';\n'.join(lines)
            fil.write(text)


def find_energy_in_text(data, loose=False, keyword=None, index=':'):
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
        s_energies: Property values as floats or dictionary of
            filename-value pairs if loose.
    """
    data = data + ';'  # ensures final value is not skipped
    if loose:
        parser = re.compile('([\w._]+)(?:[=\s,:]+)(\S+)(?:[;,\s])')
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


if __name__ == '__main__':
    description = 'Create collated molecules file.'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('--settings_file', '-s', default='settings.cfg',
                           help='Filename of settings.')
    argparser.add_argument('--verbosity', '-v', action='count')
    argparser.add_argument('--export', '-E', action='store_true',
                           help='Export entries to csv.')
    args = argparser.parse_args()
    settings = SettingsParser('Collate').read(args.settings_file)
    settings['verbosity'] = args.verbosity

    input_name = settings['inputs_name']
    output_name = settings['outputs_name']
    energies_file = settings['energies_file']
    sys_elements = settings['sys_elements']
    assert sys_elements != ['None']

    collator = BatchCollator(output_name, settings, sys_elements)
    if os.path.isfile(input_name):
        collator.parse_molecules(input_name, energies_file)
    else:
        collator.parse_loose_molecules(input_name, energies_file)
    if args.export:
        collator.export_entries(input_name.split('.')[0] + '.csv')
