"""

"""
import os
import h5py
import numpy as np
import sklearn.preprocessing as skp
from itertools import combinations_with_replacement

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


class Preprocesser():
    def __init__(self, settings):
        self.g_1 = []
        self.g_2 = []
        self.s_groups = []
        self.sys_elements = []
        self.s_energies = []
        self.s_element_counts = []
        self.index = settings['index']
        self.max_iters = settings['subsample_redistribute_iterations']
        self.size_tolerance = settings['subsample_size_tolerance']
        self.all = []
        self.bins = []
        self.subsamples = []
        self.k = settings['k-fold']
        self.slice_indices = settings['slice_indices']


    def read_fingerprints(self, filename):
        self.g_1 = []
        self.g_2 = []
        self.s_groups = []
        self.sys_elements = []
        self.s_energies = []
        self.s_element_counts = []
        with h5py.File(filename, 'r', libver='latest') as h5f:
            # read list of names from system/sys_entries dataset
            sys_entries = h5f['system']['sys_entries'][()]
            sys_entries = sys_entries[self.index]  # slice using index
            s_names = [line.split(b';')[0].decode('utf-8')
                       for line in sys_entries]
            self.s_groups = [name.split('_')[1] for name in s_names]
            self.sys_elements = [symbol.decode('utf-8')
                                 for symbol
                                 in h5f['system'].attrs['sys_elements']]
            # get master set of elements from system attributes
            s_dset = h5f['structures']  # top-level group reference
            for j, s_name in enumerate(s_names):
                # loop over structures
                dset = s_dset[s_name]  # group reference for one structure
                n_atoms = dset.attrs['natoms']
                element_set = [symbol.decode('utf-8')
                               for symbol in dset.attrs['element_set']]
                # set of elements per structure
                if not set(element_set).issubset(set(self.sys_elements)):
                    continue  # skip if not part of master set

                count_per_element = dset.attrs['element_counts']
                energy_value = float(dset.attrs['energy']) / n_atoms * 1000
                self.g_1.append(dset['G_1'][()])
                self.g_2.append(dset['G_2'][()])
                self.s_energies.append(energy_value)
                self.s_element_counts.append(count_per_element.tolist())
            print('Read {} fingerprints from {}'.format(len(self.s_energies),
                                                        filename))
            
    def subdivide_by_parameter_set(self, n_pair_params, n_triplet_params):
        assert self.g_1  # ensure read_fingerprints() completed
        pair_slice = pair_slice_choices[int(np.log2(n_pair_params))]
        triplet_slice = triplet_slice_choices[int(np.log2(n_triplet_params))]
        self.g_1 = [g[..., pair_slice] for g in self.g_1]
        self.g_2 = [g[..., triplet_slice] for g in self.g_2]

    def preprocess_fingerprints(self, n_pair_params, n_triplet_params):
        assert self.g_1  # ensure read_fingerprints() completed
        fps = [self.g_1, self.g_2]
        # get maximum occurrences per atom type
        max_per_element = np.amax(self.s_element_counts,
                                  axis=0)
        # get interactions per fingerprint
        pair_i = list(combinations_with_replacement(self.sys_elements, 1))
        triplet_i = list(combinations_with_replacement(self.sys_elements, 2))
        a = [len(pair_i), len(triplet_i)]
        # get parameters sets per fingerprint
        b = [n_pair_params, n_triplet_params]
        # rearrange data to columns by parameter set and normalize
        normalized = normalize_to_vectors(fps, a, b)

        # pad fingerprints by element
        self.all = pad_fp_by_element(normalized,
                                     self.s_element_counts,
                                     max_per_element)

    def save_preprocessed(self, filename):
        assert self.all  # ensure preprocess_fingerprints() completed
        with h5py.File(filename, 'w', libver='latest') as h5f:
            sys = h5f.require_group('system')
            sys.attrs['s_energies'] = self.s_energies
            sys.attrs['s_element_counts'] = self.s_element_counts
            sys.attrs['sys_elements'] = np.string_(self.sys_elements)
            h5f.create_dataset('preprocessed/all',
                               data=self.all, shape=self.all.shape)

    def save_stratified_kfold(self, filename):
        assert self.subsamples  # ensure preprocess_fingerprints() completed
        with h5py.File(filename, 'w', libver='latest') as h5f:
            for j, subsample in enumerate(self.subsamples):
                path = 'preprocessed/{}fold/subsample_{}'.format(self.k, j)
                h5f.create_dataset(path,
                                   data=subsample, shape=subsample.shape)

    def save_split(self, filename):
        assert self.all  # ensure preprocess_fingerprints() completed
        with h5py.File(filename, 'w', libver='latest') as h5f:
            h5f.create_dataset('preprocessed/',
                               data=self.all, shape=self.all.shape)

    def stratified_kfold_bins(self):
        '''
        with h5py.File(filename, 'r',
                       libver='latest') as h5f:
    
            sys_entries = h5f['system']['sys_entries'][()]
            s_names = [line.split(b';')[0].decode('utf-8')
                       for line in sys_entries]
            s_groups = [name.split('_')[1] for name in s_names]
            s_element_counts = [line.split(b';')[3].decode('utf-8').split(',')
                       for line in sys_entries]
            s_energies = [float(line.split(b';')[-1])
                          for line in sys_entries]
        '''
        assert self.s_groups  # ensure read_fingerprints() completed
        comps = {group: comp
                 for group, comp in zip(self.s_groups, 
                                        self.s_element_counts)}
    
        energy_dict = {group: energy
                       for group, energy in zip(self.s_groups, 
                                                self.s_energies)}
    
        groups_set = sorted(list(set(self.s_groups)), key=comps.get)
    
        bins = [groups_set[i::self.k] for i in range(self.k)]
    
        print([len(x) for x in bins])
    
        name_counts = [self.s_groups.count(x) for x in groups_set]
    
        bins_tot = [np.sum([name_counts[groups_set.index(entry)]
                            for entry in bi]) for bi in bins]
        print(bins_tot)
    
        iteration = 0
        size_tolerance = np.mean(bins_tot) / self.size_tolerance
        energy_best = np.inf
        bins_best = list(bins)
    
        while iteration < self.max_iters:
            iteration += 1
            smallest = np.argmin(bins_tot)
            largest = np.argmax(bins_tot)
            rand_ind = int(
                    np.round(np.abs(np.random.rand()
                                    * (len(bins[largest]) / 2))))
            bins[smallest].append(bins[largest].pop(rand_ind))
            bins_tot = [np.sum([name_counts[groups_set.index(entry)]
                                for entry in bi]) for bi in bins]
            mean_energies = [mean_energy(energy_dict,
                                         name_counts,
                                         groups_set,
                                         entries) for entries in bins]
            if (np.std(bins_tot) < size_tolerance
                    and np.std(mean_energies) < energy_best):
                energy_best = np.std(mean_energies)
                bins_best = bins
        self.bins = bins_best


def mean_energy(energy_dict, name_counts, groups_set, entries):
    """
    Weighted average of energy. Intermediate function for
    sorting stratified kfold chunks.
    """
    products = [np.multiply(energy_dict[entry],
                            name_counts[groups_set.index(entry)])
                for entry in entries]
    divisor = np.sum([name_counts[groups_set.index(entry)]
                      for entry in entries])
    return np.sum(products) / divisor


def normalize_to_vectors(unprocessed_data, Alpha, Beta):
    """

    Args:
        unprocessed_data: List of data sets (e.g. G1, G2)
            with equal-size first dimension.
        Alpha: interactions w.c. per data set
        Beta: parameter sets per data set

    Returns:
        processed_data: Flattened, normalized dataset.
            Length = per structure,
            width = feature vector (i.e. flattened and concatenated G1 and G2).

    """

    vectorized_fp_list = []
    for dataset, alpha, beta in zip(unprocessed_data, Alpha, Beta):
        fp_columns = [rearrange_to_columns(fp_per_s)
                      for fp_per_s in dataset]
        fp_lengths = [len(fingerprint) for fingerprint in
                      fp_columns]
        fp_as_grid = np.concatenate(fp_columns)

        normalized_grid = skp.normalize(fp_as_grid, axis=0, norm='max')

        print('alpha={}, beta={}'.format(alpha, beta),
              '  min=', np.amin(normalized_grid),
              '  mean=', np.mean(normalized_grid),
              '  std=', np.std(normalized_grid),
              '  max=', np.amax(normalized_grid))

        regenerated_fps = regenerate_fp_by_len(normalized_grid, fp_lengths)
        vectorized_fp_list.append(vectorize_fps(regenerated_fps, alpha))
    normalized_fps = [np.concatenate((g1t, g2t), axis=1) for g1t, g2t
                      in zip(*vectorized_fp_list)]

    return normalized_fps
    # for each fingerprint,
    # length = # structures * # combinations w.r. of interactions
    # shape = ((S * j), k)

    # regenerate data per structure by slicing based on recorded atoms
    # per structure and then flatten for one 1d vector per structure


def rearrange_to_columns(fingerprint):
    layers = [layer for layer in np.asarray(fingerprint)]
    columns = np.concatenate(layers, axis=0)
    return columns


def regenerate_fp_by_len(fp_as_grid, fp_lengths):
    indices = np.cumsum(fp_lengths)[:-1]
    regenerated_fps = np.split(fp_as_grid, indices)
    return regenerated_fps


def vectorize_fps(regenerated_fps, alpha):
    lengths = [len(fingerprint) for fingerprint in regenerated_fps]
    vectors = [[atom.flatten(order='F')
                for atom in np.split(grid,
                                     np.arange(alpha, length, alpha))]
               for grid, length in zip(regenerated_fps,
                                       lengths)]
    return vectors


def pad_fp_by_element(input_data, compositions, final_layers):
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
            in the 2nd dimension (equal number of atoms per structure).

    """
    new_input_data = []
    for structure, initial_layers in zip(input_data, compositions):
        assert len(final_layers) == len(initial_layers)
        data = np.asarray(structure)
        data_shape = data.shape
        natoms_i = data_shape[0]
        natoms_f = sum(final_layers)
        natoms_diff = natoms_f - natoms_i
        secondary_dims = len(data_shape) - 1
        pad_widths = [(natoms_diff, 0)] + [(0, 0)] * secondary_dims
        # tuple of (header_pad, footer_pad) per dimension
        # if masking:
        data_f = np.pad(np.ones(data.shape), pad_widths, 'edge') * 0.0
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


def read_preprocessed(filename, k_test_ind, split_positions,
                      n_pair_params, n_trip_params):
    with h5py.File(filename, 'r', libver='latest') as h5f:
        sys = h5f['system']
        energy_list = sys.attrs['s_energies']
        element_counts_list = sys.attrs['s_element_counts']
        sys_elements = [x.decode() for x in sys.attrs['sys_elements']]
        padded = h5f['{}-{}'.format(n_pair_params, n_trip_params)][()]

    n_list = np.sum(element_counts_list, axis=1)

    max_per_element = np.amax(element_counts_list,
                              axis=0)  # maximum occurrences per atom type

    assert not np.any(np.isnan(padded))
    assert not np.any(np.isinf(padded))

    chunks = np.split(padded, split_positions)
    valid_inputs = chunks.pop(k_test_ind).transpose([1, 0, 2])
    train_inputs = np.concatenate(chunks).transpose([1, 0, 2])
    print(train_inputs.shape[1], valid_inputs.shape[1])

    chunked_names = np.split(n_list, split_positions)
    n_list_valid = chunked_names.pop(k_test_ind)
    n_list_train = np.concatenate(chunked_names)

    print(len(n_list_train), len(n_list_valid))

    chunked_energy = np.split(energy_list, split_positions)
    valid_outputs = chunked_energy.pop(k_test_ind)
    train_outputs = np.concatenate(chunked_energy)

    print(len(train_outputs), len(valid_outputs))

    chunked_compositions = np.split(element_counts_list, split_positions)
    valid_compositions = chunked_compositions.pop(k_test_ind)
    valid_compositions = [','.join([str(x) for x in y])
                          for y in valid_compositions]
    print(set(valid_compositions))

    return (train_inputs, valid_inputs,
            train_outputs, valid_outputs,
            sys_elements, max_per_element, n_list_train, n_list_valid)
