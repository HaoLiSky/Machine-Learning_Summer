"""
This module provides input/output functions for crystal/molecule data
(e.g. from VASP outputs).
"""
import os
import re
import h5py
import traceback
import numpy as np
from ase.io import iread
from nnf.io_utils import slice_from_str, write_to_group, read_from_group


class Collator:
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
        sys_elements: List of system-wide unique element names as strings.
    """

    def __init__(self, output_name, sys_elements):
        self.output_name = output_name
        self.sys_elements = sys_elements
        self.element_order = {k: v for v, k in enumerate(sys_elements)}
        self.entries = []
        self.sources = []
        self.m_count = 0

    def parse_molecules(self, filename, energy_file, keyword=None,
                        index=':', form=None):
        """
        Args:
            filename (str): Filename of crystal/molecule data.
            energy_file: List of property values as floats.
            keyword (str): Optional string to find property values.
                (e.g "energy" to find "energy: 0.423eV"
                or "E0" to find "E0= -0.52E+02")
            index (str): Flexible indexing.
            form (str): Optional file format string to
                pass to ASE's read function.
        """
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
        with h5py.File(self.output_name, 'a', libver='latest') as h5f:
            structures = iread(filename, format=form, index=index)
            for structure in structures:
                m_name = '{}.{}'.format(filename, self.m_count)
                try:
                    energy_val = energy_data[self.m_count - m_count_start]
                    data = parse_molecule(structure, m_name,
                                          energy_val, h5f,
                                          self.sys_elements,
                                          self.element_order)
                    self.entries.append(data)
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

    def parse_loose_molecules(self, directory, energy_file, index=':',
                              form=None):
        """
        Args:
            directory (str): Root directory of filetree of
                crystal/molecule data.
            energy_file: Dictionary of filename-value pairs.
            index (str): Flexible indexing.
            form (str): Optional file format string to
                pass to ASE's read function.
        """
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
        with h5py.File(self.output_name, 'a', libver='latest') as h5f:
            for root, dirs, files in os.walk(directory):
                sorted_files = sorted(files)
                for filename in sorted_files[slice_from_str(index)]:
                    try:
                        structures = iread(os.path.join(root, filename),
                                           format=form)
                    except (ValueError, IOError, IndexError):
                        continue
                    for structure in structures:
                        m_name = '{}.{}'.format(filename, self.m_count)
                        try:
                            energy_val = energy_data[filename]
                            data = parse_molecule(structure, m_name,
                                                  energy_val, h5f,
                                                  self.sys_elements,
                                                  self.element_order)
                            self.entries.append(data)
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
        with h5py.File(filename, 'a', libver='latest') as h5i:
            m_count = len(h5i.require_group('molecules').keys())
            m_count_start = int(m_count)
            print(m_count_start, 'collated molecules in file.')
            try:
                entries_list = h5i['system/collated_entries'][()].tolist()
                self.entries.extend([line.split(b';')
                                     for line in entries_list])
                sources_list = h5i['system/collated_sources'][()].tolist()
                self.sources.extend([line.split(b';')
                                     for line in sources_list])
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
                        m_count += 1
                print(m_count - m_count_start, ' added.')
                self.m_count += m_count
                self.update_system_details(h5i)

    def update_system_details(self, h5f):
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
        print('Entries:', len(self.entries))
        sizes = [int(entry[1]) for entry in self.entries]
        compositions = [entry[3].decode('utf-8') for entry in self.entries]
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
        header = 'Name,Size,Elements,Composition,Coords Shape,Energy;\n'
        with open(filename, 'w') as fil:
            lines = [b','.join(line).decode('utf-8')
                     for line in self.entries]
            text = header + ';\n'.join(lines)
            fil.write(text)


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

    def __init__(self, ):
        self.coords = []
        self.element_set = []
        self.element_counts = []
        self.elements_list = []
        self.unit_vectors = None
        self.periodic = None
        self.energy_val = 0.0
        self.attrs_dict = {}
        self.dsets_dict = {}

    def from_file(self, h5f, path, sys_elements):
        """
        Args:
            h5f: h5py object for reading.
            path (str): Path in h5f to molecule.
            sys_elements: List of system-wide unique element names as strings.
        """
        self.attrs_dict, self.dsets_dict = read_from_group(h5f, path)

        self.coords = self.dsets_dict['coordinates']
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

    def to_file(self, h5o, path):
        """
        Args:
            h5o: h5py object for writing.
            path (str): Path in h5o to molecule.
        """
        write_to_group(h5o, path, self.attrs_dict, self.dsets_dict)


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


def parse_molecule(structure, m_name, energy_val,
                   h5f, sys_elements, element_order):
    """
    Writes one crystal/molecule object's data to .hdf5 file.

    Args:
        h5f: h5py object for writing.
        structure: ASE Atoms object.
        sys_elements: List of system-wide unique element names as strings.
        element_order: Dictionary of elements and specified order.
        m_name (str): Atoms' identifier to use as group name.
        energy_val (float): Atoms' corresponding property value.
    """
    coords = structure.get_positions(wrap=False)
    natoms = len(coords)
    elements = structure.get_chemical_symbols()
    elements, coords = zip(*sorted(zip(elements, coords),
                                   key=lambda x: element_order.get(x[0])))
    # ensure that molecule belongs in system
    element_set = sorted(list(set(elements)),
                         key=element_order.get)
    assert set(element_set).issubset(set(sys_elements))
    # ensure that coordinate array size matches composition
    element_counts = np.asarray([elements.count(element)
                                 for element
                                 in sys_elements]).astype('i4')
    assert sum(element_counts) == natoms

    try:
        unit_vectors = structure.get_cell(complete=False)
        periodic = True
    except (ValueError, RuntimeError, AttributeError):
        unit_vectors = np.zeros((3, 3))
        periodic = False

    if np.all(np.isclose(unit_vectors, np.zeros((3, 3)))):
        periodic = False

    dict_dsets = {'coordinates': coords}
    dict_attrs = {'natoms'        : natoms,
                  'element_set'   : np.string_(element_set),
                  'element_counts': element_counts,
                  'unit_vectors'  : unit_vectors,
                  'periodic'      : periodic,
                  'energy'        : energy_val}

    group_name = 'structures/{}'.format(m_name)
    write_to_group(h5f, group_name, dict_attrs, dict_dsets)
    data = [np.string_(m_name),
            np.string_(str(natoms)),
            np.string_(','.join(element_set)),
            np.string_(','.join([str(x) for x in element_counts])),
            np.string_(str(periodic)),
            np.string_(str(coords.shape)),
            np.string_(energy_val)]
    return data
