"""
Preprocessing fingerprints for network input.
"""
import os
import argparse
import h5py
import numpy as np
import sklearn.preprocessing as skp
from itertools import combinations_with_replacement
from nnf.io_utils import slice_from_str, read_from_group, write_to_group
from nnf.io_utils import SettingsParser
from nnf.io_utils import grid_string
from nnf.batch_fingerprint import Fingerprint

np.random.seed(8)
pair_slice_choices = [[2],
                      [2, 6],
                      [0, 2, 4, 6],
                      slice(None, None, None)]
triplet_slice_choices = [[8],
                         [8, 9],
                         [8, 9, 26, 27],
                         [0, 1, 8, 9, 18, 19, 26, 27],
                         [0, 1, 4, 5, 8, 9, 12, 13, 18, 19, 22, 23, 26,
                          27, 30, 31],
                         slice(None, None, None)]


class DataPreprocessor:
    """
    Preprocess fingerprint data for neural network training.
    """

    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)

    def read_fingerprints(self, filename):
        """
        Load fingerprint data and system details.

        Args:
            filename: Fingerprints file (e.g. fingerprints.h5).
        """
        self.g = []
        self.dg = []
        self.m_names = []
        self.m_groups = []
        self.sys_elements = []
        self.m_energies = []
        self.m_element_counts = []
        self.standards = []

        libver = self.settings['libver']
        index = self.settings['index']
        with h5py.File(filename, 'r', libver=libver) as h5f:
            # 1) read list of names from system/sys_entries dataset
            self.m_names = list(h5f.require_group('fingerprints').keys())
            self.m_names = np.asarray(self.m_names)[slice_from_str(index)]
            self.sys_elements = [symbol.decode('utf-8')
                                 for symbol
                                 in h5f['system'].attrs['sys_elements']]
            # 2) Loop through fingerprints, loading data to object
            for j, m_name in enumerate(self.m_names):
                print('read', j, end='\r')
                path = 'fingerprints/' + m_name
                fp = Fingerprint()
                fp.from_file(h5f, path, self.sys_elements)
                self.g.append([fp.dsets_dict[key]
                               for key in fp.dsets_dict.keys()
                               if key.find('G_') == 0])
                self.dg.append([fp.dsets_dict[key]
                                for key in fp.dsets_dict.keys()
                                if key.find('dG_') == 0])
                energy = float(fp.energy_val) * 1000 / fp.natoms
                # energy per atom
                self.m_energies.append(energy)
                self.m_element_counts.append(fp.element_counts)
                if not set(fp.elements_set).issubset(set(self.sys_elements)):
                    continue  # skip if not part of system
            self.g = [list(g) for g in zip(*self.g)]
            self.dg = [list(dg) for dg in zip(*self.dg)]
            print('Read {} fingerprints from {}'.format(len(self.m_energies),
                                                        filename))

    def subdivide_by_parameter_set(self, divisions_by_fingerprint):
        """
        Slice fingerprints by final dimension (parameter sets).

        Args:
            divisions_by_fingerprint: List of desired parameter set lengths
                per fingerprint (e.g. [8,16] for 8 pair parameter sets and
                16 triplet parameter sets.)
        """
        assert self.g  # ensure read_fingerprints() completed

        # temp; hardcoded for g1 and g2
        n_pair_params, n_triplet_params = divisions_by_fingerprint
        pair_slice = pair_slice_choices[int(np.log2(n_pair_params))]
        triplet_slice = triplet_slice_choices[int(np.log2(n_triplet_params))]
        index_list = [pair_slice, triplet_slice]
        # temp; hardcoded for g1 and g2

        for j, indexing in enumerate(index_list):
            self.g[j] = [g[..., indexing] for g in self.g[j]]

    def preprocess_fingerprints(self):
        """
        Preprocess fingerprints.
        """
        assert self.g  # ensure read_fingerprints() completed
        self.pad_val = self.settings['padding_value']
        # 1) get maximum occurrences per atom type
        max_per_element = np.amax(self.m_element_counts, axis=0)
        # 2) get interactions per fingerprint
        pair_i = list(combinations_with_replacement(self.sys_elements, 1))
        triplet_i = list(combinations_with_replacement(self.sys_elements, 2))
        alpha = [len(pair_i), len(triplet_i)]
        # 3) normalize
        normalized, self.standards = normalize_to_vectors(self.g, alpha)
        # 4) pad fingerprints by element
        self.all = pad_fp_by_element(normalized,
                                     self.m_element_counts,
                                     max_per_element,
                                     pad_val=self.pad_val)

    def to_file(self, filename):
        """
        Write preprocessed fingerprints to file.

        Args:
            filename (str): Output file (e.g. preprocessed.h5).
        """
        assert self.standards  # ensure preprocess_fingerprints() completed
        libver = self.settings['libver']
        with h5py.File(filename, 'w', libver=libver) as h5f:
            energies = np.asarray(self.m_energies)
            element_counts = np.asarray(self.m_element_counts)
            m_names = np.string_(self.m_names.tolist())
            write_to_group(h5f, 'preprocessed',
                           {},
                           {'energies'      : energies,
                            'element_counts': element_counts,
                            'all'           : self.all,
                            'm_names'       : m_names},
                           dset_types={'energies'      : energies.dtype,
                                       'element_counts': element_counts.dtype,
                                       'all'           : self.all.dtype,
                                       'm_names'       : m_names.dtype})

            scaling_standards = {'standard_{}'.format(j): standard
                                 for j, standard in enumerate(self.standards)}
            write_to_group(h5f, 'system',
                           {'sys_elements': np.string_(self.sys_elements)},
                           scaling_standards)

    def generate_simple_partitions(self, filename, split_frac, k=0):
        """
        Generate comma-separated value file with training/testing
        designations for each fingerprint.

        Args:
            filename (str): Filename of .csv (e.g. partitions.csv)
            split_frac (float): Fraction of dataset to designate for training.
                e.g. 0.5 to set aside half for testing and half for training
            k (int): Optional number of subsamples for stratified k-fold.
                e.g. 10 for 10-fold cross validation
        """
        assert k > 1 or split_frac < 1
        assert self.m_energies
        m_names = self.m_names
        np.random.shuffle(m_names)
        bins_dict = {}
        # 1) separate data for testing set using specified fraction
        if split_frac < 1:
            split_ind = [int(np.ceil(len(m_names) * split_frac))]
            training_group, validation_group = np.split(m_names, split_ind)
            for entry in validation_group:
                bins_dict[entry] = -1
        else:
            training_group = m_names
        # 2) generate subsamples for stratified k-fold cross validation
        if k > 1:
            bins = [training_group[i::k] for i in range(k)]
            for j, bin_ in enumerate(bins):
                for entry in bin_:
                    bins_dict[entry] = j
        else:
            for entry in training_group:
                bins_dict[entry] = 0
        organized_entries = sorted([(m_name, str(bins_dict[m_name]))
                                    for m_name in m_names],
                                   key=lambda x: x[1])
        # 3) Write to file
        header = 'Name,Designation (-1: testing, 0+: training);\n'
        lines = [','.join(pair) for pair in organized_entries]
        with open(filename, 'w') as fil:
            text = header + ';\n'.join(lines)
            fil.write(text)

    def generate_partitions(self, filename, split_frac, k=0):
        """
        Generate comma-separated value file with training/testing
        designations for each fingerprint. Groups are defined by the second-
        to-last number in

        Args:
            filename (str): Filename of .csv (e.g. partitions.csv)
            split_frac (float): Fraction of dataset to designate for training.
                e.g. 0.5 to set aside half for testing and half for training
            k (int): Optional number of subsamples for stratified k-fold.
                e.g. 10 for 10-fold cross validation
        """
        max_iters = self.settings['max_iters']
        size_tolerance_factor = self.settings['size_tolerance_factor']
        assert self.m_energies  # ensure read_fingerprints() completed
        assert k > 1 or split_frac < 1
        self.m_groups = [name.split('.')[0].split('_')[-2]
                         for name in self.m_names]
        comps = {group: str(comp) for group, comp
                 in zip(self.m_groups, self.m_element_counts)}
        energy_dict = {group: energy for group, energy
                       in zip(self.m_groups, self.m_energies)}
        groups_set = sorted(list(set(self.m_groups)), key=comps.get)
        size_dict = {group: self.m_groups.count(group) for group in groups_set}
        bins_dict = {}
        # 1) separate data for testing set using specified fraction
        if split_frac < 1:
            iteration = 0
            total_entries = len(self.m_names)
            validation_groups = []
            validation_total = 0
            while iteration < max_iters:
                iteration += 1
                # randomly choose a data group
                choice = np.random.choice(groups_set)
                new_frac = ((validation_total + size_dict[choice])
                            / total_entries)
                # designate to testing set if it fits
                if new_frac <= 1 - split_frac:
                    validation_total += size_dict[choice]
                    ind = groups_set.index(choice)
                    validation_groups.append(groups_set.pop(ind))
                    if new_frac == split_frac:
                        break
                        # if testing set size matches desired size, stop early
            for entry in validation_groups:
                bins_dict[entry] = -1
        # 2) generate subsamples for stratified k-fold cross validation
        if k > 1:
            bins = [groups_set[i::k] for i in range(k)]
            bin_sizes = [np.sum([size_dict[entry] for entry in bin_])
                         for bin_ in bins]
            print(bin_sizes)

            iteration = 0
            size_tolerance = np.mean(bin_sizes) / size_tolerance_factor
            energy_best = np.inf
            bins_best = list(bins)
            while iteration < max_iters:
                iteration += 1
                bins = [sorted(bin_, key=size_dict.get) for bin_ in bins]
                # bins internally sorted from smallest groups to largest groups
                smallest = int(np.argmin(bin_sizes))
                largest = int(np.argmax(bin_sizes))
                rand_ind = int(np.round(np.abs(np.random.rand()
                                               * (len(bins[largest]) / 2))))
                # randomly choose group from first half of largest bin
                bins[smallest].append(bins[largest].pop(rand_ind))
                bin_sizes = [np.sum([size_dict[entry]
                                     for entry in bin_]) for bin_ in bins]
                mean_energies = [mean_energy(bin_,
                                             energy_dict,
                                             size_dict) for bin_ in bins]
                if (np.std(bin_sizes) < size_tolerance
                        and np.std(mean_energies) < energy_best):
                    energy_best = np.std(mean_energies)
                    bins_best = bins
            bin_sizes = [np.sum([size_dict[entry] for entry in bin_])
                         for bin_ in bins_best]
            print(bin_sizes)
            for j, bin_ in enumerate(bins_best):
                for entry in bin_:
                    bins_dict[entry] = j
        else:  # i.e. only split into training and testing groups
            for entry in groups_set:
                bins_dict[entry] = 0
        # 3) Assign designation tags to each molecule's fingerprint
        organized_entries = []
        for m_name in self.m_names:
            group = m_name.split('.')[0].split('_')[-2]
            bin_designation = bins_dict[group]
            organized_entries.append([m_name, str(bin_designation)])
        organized_entries = sorted(organized_entries, key=lambda x: x[1])
        # 4) Write to file
        header = 'Name,Designation (-1: testing, 0+: training);\n'
        lines = [','.join(pair) for pair in organized_entries]
        with open(filename, 'w') as fil:
            text = header + ';\n'.join(lines)
            fil.write(text)


def mean_energy(bin_, energy_dict, size_dict):
    """
    Weighted average of energy. Intermediate function for
    sorting stratified kfold chunks.
    """
    products = [np.multiply(energy_dict[entry],
                            size_dict[entry]) for entry in bin_]
    divisor = np.sum([size_dict[entry] for entry in bin_])
    return np.sum(products) / divisor


def read_partitions(filename):
    """
    Args:
        filename: File with training/testing partitions (e.g. partitions.csv)

    Returns:
        Dictionary of molecule ids and their partition designations.
    """
    with open(filename, 'r') as fil:
        lines = fil.readlines()[1:]  # skip header
    entries = [line.replace(';\n', '').split(',') for line in lines]
    partition_dict = {m_name: int(bin_) for m_name, bin_ in entries}
    return partition_dict


def load_preprocessed(filename, partitions_file, k_test=0,
                      validation_tag=-1, libver='latest',
                      verbosity=1):
    """
    Args:
        filename (str): File with preprocessed fingerprint data.
            e.g. fingerprints.h5
        partitions_file (str): File with training/testing partitions.
            e.g. partitions.csv
        k_test (int): Optional index of subsample for testing with
            stratified k-fold cross validation.
        validation_tag: Optional designation tag for validation data (i.e.
            not introduced to the network as either training or testing).
                Defaults to -1.
        libver (str): Optional h5py argument for i/o. Defaults to 'latest'.
        verbosity (int): Print details if greater than 0.

    Returns:
        system (dict): System details.
        training (dict): Data partitioned for training.
        testing (dict): Data partitioned for testing.
    """
    # 1) Read data from hdf5 file
    with h5py.File(filename, 'r', libver=libver) as h5f:
        attrs_dict, dsets_dict = read_from_group(h5f, 'preprocessed')
        sys_elements = h5f['system'].attrs['sys_elements']
        m_names = [entry.decode('utf-8')
                   for entry in dsets_dict['m_names']]
        energies = dsets_dict['energies']
        compositions = dsets_dict['element_counts']
        all_data = dsets_dict['all']
    if os.path.isfile('ignore_tags'):
        with open('ignore_tags', 'r') as file_:
            ignore_tags = file_.read().split('\n')
            ignore_tags = list(filter(None, ignore_tags))
    else:
        ignore_tags = []
    partitions = read_partitions(partitions_file)
    sizes = np.sum(compositions, axis=1)
    max_per_element = np.amax(compositions, axis=0)
    # maximum occurrences per atom type
    assert not np.any(np.isnan(all_data))
    assert not np.any(np.isinf(all_data))

    training_set = []
    testing_set = []
    validation_set = []
    # 2) Get designation tags for each molecule's fingerprint
    for j, m_name in enumerate(m_names):
        if np.any([ignore in m_name for ignore in ignore_tags]):
            pass
        elif partitions[m_name] == k_test:
            testing_set.append(j)
        elif partitions[m_name] == validation_tag:
            validation_set.append(j)
        else:
            training_set.append(j)

    # 3) Distribute entries to training or testing sets accordingly
    training = {}
    testing = {}
    training['inputs'] = np.take(all_data, training_set, axis=0)
    testing['inputs'] = np.take(all_data, testing_set, axis=0)
    train_samples, in_neurons, feature_length = training['inputs'].shape
    test_samples = len(testing['inputs'])

    training['inputs'] = training['inputs'].transpose([1, 0, 2])
    testing['inputs'] = testing['inputs'].transpose([1, 0, 2])

    training['outputs'] = np.take(energies, training_set)
    testing['outputs'] = np.take(energies, testing_set)

    comp_strings = ['-'.join([str(el) for el in com])
                    for com in compositions]
    training['compositions'] = np.take(comp_strings, training_set, axis=0)
    testing['compositions'] = np.take(comp_strings, testing_set, axis=0)

    train_c = set(training['compositions'])
    test_c = set(testing['compositions'])
    common = grid_string(sorted(train_c.intersection(test_c)))
    train_u = grid_string(sorted(train_c.difference(test_c)))
    test_u = grid_string(sorted(test_c.difference(train_c)))

    if verbosity > 0:
        print('\nTraining samples:', str(train_samples).ljust(15),
              'Testing samples:', test_samples)
        print('Input neurons (max atoms):', str(in_neurons).ljust(15),
              'Feature-vector length:', feature_length)
        print('Unique compositions in training set:')
        print(train_u)
        print('Unique compositions in testing set:')
        print(test_u)
        print('Shared compositions in both training and testing sets:')
        print(common)

    training['sizes'] = np.take(sizes, training_set)
    testing['sizes'] = np.take(sizes, testing_set)

    training['names'] = np.take(m_names, training_set, axis=0)
    testing['names'] = np.take(m_names, testing_set, axis=0)

    system = {'sys_elements'   : sys_elements,
              'max_per_element': max_per_element}
    return system, training, testing


def normalize_to_vectors(unprocessed_data, Alpha):
    """
    Args:
        unprocessed_data: List of data sets (e.g. G1, G2)
            with equal-size first dimension.
        Alpha: interactions w.c. per data set
            e.g. [2, 3] for a binary system

    Returns:
        processed_data: Flattened, normalized dataset.
        Length = per structure,
        width = feature vector (i.e. flattened and concatenated G1 and G2).
    """
    vectorized_fp_list = []
    standards = []
    for dataset, alpha in zip(unprocessed_data, Alpha):
        # 1) Rearrange to columns sharing last dimension
        fp_columns = [rearrange_to_columns(fp_per_s)
                      for fp_per_s in dataset]
        fp_lengths = [len(fingerprint) for fingerprint in
                      fp_columns]
        # 2) Join into grid
        fp_as_grid = np.concatenate(fp_columns)
        # 3) Normalize along column-axis using scikit-learn
        st_min = np.max(fp_as_grid, axis=0)
        st_max = np.min(fp_as_grid, axis=0)
        st_mean = np.mean(fp_as_grid, axis=0)
        standards.append(np.stack([st_min, st_max, st_mean], axis=0))
        normalized_grid = skp.normalize(fp_as_grid, axis=0, norm='max')

        print('alpha={}'.format(alpha),
              '  min=', np.amin(normalized_grid),
              '  mean=', np.mean(normalized_grid),
              '  std=', np.std(normalized_grid),
              '  max=', np.amax(normalized_grid))
        # 4) Regenerate fingerprints per molecule
        regenerated_fps = regenerate_fp_by_len(normalized_grid, fp_lengths)
        # 5) Flatten fingerprints to vectors
        vectorized_fp_list.append(vectorize_fps(regenerated_fps, alpha))
    # 6) Concatenate all fingerprints for each molecule
    normalized_fps = [np.concatenate(fps, axis=1) for fps
                      in zip(*vectorized_fp_list)]
    return normalized_fps, standards


def rearrange_to_columns(fingerprint):
    """
    Args:
        fingerprint: Multidimensional numpy array.

    Returns:
        Concatenated fingerprint.
    """
    layers = [layer for layer in np.asarray(fingerprint)]
    columns = np.concatenate(layers, axis=0)
    return columns


def regenerate_fp_by_len(fp_as_grid, fp_lengths):
    """
    Args:
        fp_as_grid: Fingerprint data as indistinct grid.
        fp_lengths: List of lengths of fingerprint vectors.

    Returns:
        Regenerated fingerprints as list of numpy arrays.
    """
    indices = np.cumsum(fp_lengths)[:-1]
    regenerated_fps = np.split(fp_as_grid, indices)
    return regenerated_fps


def vectorize_fps(regenerated_fps, alpha):
    """
    Args:
        regenerated_fps: list of fingerprints as numpy arrays.
        alpha: length of last dimension in fingerprint arrays.

    Returns:
        List of flattened fingerprint vectors.
    """
    lengths = [len(fingerprint) for fingerprint in regenerated_fps]
    vectors = [[atom.flatten(order='F')
                for atom in np.split(grid,
                                     np.arange(alpha, length, alpha))]
               for grid, length in zip(regenerated_fps,
                                       lengths)]
    return vectors


def pad_fp_by_element(input_data, compositions, final_layers, pad_val=0.0):
    """
    Slice fingerprint arrays by elemental composition and
    arrange slices onto empty* arrays to produce sets of padded fingerprints
    (i.e. equal length for each type of fingerprint)

    *padding is set to a constant -1, to be used in Keras' masking function.

    Args:
        input_data: 2D array of structures/features.
            i.e. output of normalize_inputs()
        compositions: list of "layer" heights (per element) in # of atoms.
        final_layers: list of desired "layer" heights (per element)
            in # of atoms.
    Returns:
        output_data: 2D array of structures/features, padded to equal lengths
            in the 2nd dimension (equal number of atoms per molecule).
    """
    new_input_data = []
    for molecule, initial_layers in zip(input_data, compositions):
        assert len(final_layers) == len(initial_layers)
        data = np.asarray(molecule)
        data_shape = data.shape
        natoms_i = data_shape[0]
        natoms_f = sum(final_layers)
        natoms_diff = natoms_f - natoms_i
        secondary_dims = len(data_shape) - 1
        pad_widths = [(natoms_diff, 0)] + [(0, 0)] * secondary_dims
        # tuple of (header_pad, footer_pad) per dimension
        # if masking:
        data_f = np.pad(np.ones(data.shape), pad_widths, 'edge') * pad_val
        # else:
        # data_f = np.pad(np.zeros(data.shape), pad_widths, 'edge')
        slice_pos = np.cumsum(initial_layers)[:-1]
        # row indices to slice to create n sections
        data_sliced = np.split(data, slice_pos)

        start_pos = np.insert(np.cumsum(final_layers)[:-1], 0, 0)
        # row indices to place sections of correct length

        for sect, start in zip(data_sliced, start_pos):
            end = start + len(sect)
            data_f[start:end, ...] = sect
        new_input_data.append(np.asarray(data_f))
    return np.asarray(new_input_data)


if __name__ == '__main__':
    description = 'Create preprocessed data file.'
    argparser = argparse.ArgumentParser(description=description)
    argparser.add_argument('--settings_file', '-s', default='settings.cfg',
                           help='Filename of settings.')
    argparser.add_argument('--verbosity', '-v', default=0,
                           action='count')
    argparser.add_argument('--export', '-E', action='store_true',
                           help='Export entries to csv.')
    args = argparser.parse_args()
    settings = SettingsParser('Preprocess').read(args.settings_file)
    settings['verbosity'] = args.verbosity

    input_name = settings['inputs_name']
    output_name = settings['outputs_name']
    sys_elements = settings['sys_elements']
    partitions_file = settings['partitions_file']
    split_fraction = settings['split_fraction']
    kfold = settings['kfold']
    assert sys_elements != ['None']

    preprocessor = DataPreprocessor(settings)
    preprocessor.read_fingerprints(input_name)
    subdivisions = [int(val) for val in settings['subdivisions']]
    preprocessor.subdivide_by_parameter_set(subdivisions)
    preprocessor.preprocess_fingerprints()
    preprocessor.to_file(output_name)
    preprocessor.generate_partitions(partitions_file, split_fraction, k=kfold)
