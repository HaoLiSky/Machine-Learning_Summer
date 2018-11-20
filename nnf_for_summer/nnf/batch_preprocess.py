"""
Preprocessing fingerprints for network input.
"""
import os
import argparse
import h5py
import numpy as np
#import sklearn.preprocessing as skp
#import matplotlib.pyplot as plt
#import matplotlib.ticker as ticker
#from matplotlib.colors import LogNorm
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
        descriptor = self.settings['descriptor']
        with h5py.File(filename, 'r', libver=libver) as h5f:
            # 1) read list of names from system/sys_entries dataset
            self.m_names = list(h5f.require_group(descriptor).keys())
            self.m_names = np.asarray(self.m_names)[slice_from_str(index)]
            self.sys_elements = [symbol.decode('utf-8')
                                 for symbol
                                 in h5f['system'].attrs['sys_elements']]
            # 2) Loop through fingerprints, loading data to object
            for j, m_name in enumerate(self.m_names):
                print('read', j, end='\r')
                path = descriptor + '/' + m_name
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


class PartitionProcessor:
    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)

    def load_preprocessed(self, filename, libver='latest'):
        """
        Args:
            filename (str): File with preprocessed fingerprint data.
                e.g. preprocessed.h5
            libver (str): Optional h5py argument for i/o. Defaults to 'latest'.
        """
        # 1) Read data from hdf5 file
        with h5py.File(filename, 'r', libver=libver) as h5f:
            attrs_dict, dsets_dict = read_from_group(h5f, 'preprocessed')
            self.sys_elements = [specie.decode('utf-8')
                                 for specie in
                                 h5f['system'].attrs['sys_elements']]
            self.m_names = [entry.decode('utf-8')
                            for entry in dsets_dict['m_names']]
            self.m_energies = dsets_dict['energies']
            self.compositions = dsets_dict['element_counts']
            self.all_data = dsets_dict['all']
        if os.path.isfile('ignore_tags'):
            with open('ignore_tags', 'r') as file_:
                ignore_tags = file_.read().split('\n')
                self.ignore_tags = list(filter(None, ignore_tags))
        else:
            self.ignore_tags = []

        self.sizes = np.sum(self.compositions, axis=1)
        self.max_per_element = np.amax(self.compositions, axis=0)
        # maximum occurrences per atom type
        assert not np.any(np.isnan(self.all_data))
        assert not np.any(np.isinf(self.all_data))

    def load_partitions_from_file(self, filename):
        """
        Class wrapper for read_partitions().
        """
        self.part_dict = read_partitions(filename)

    def get_network_inputs(self, testing_tags=(0), verbosity=1):
        """
         Args:
             testing_tag (int): Designation tag for testing.
             verbosity (int): Print details if greater than 0.

         Returns:
             system (dict): System details.
             training (dict): Data partitioned for training.
             testing (dict): Data partitioned for testing.
         """
        training_set = []
        testing_set = []
        # 2) Get designation tags for each molecule's fingerprint
        for j, m_name in enumerate(self.m_names):
            if np.any([ignore in m_name for ignore in self.ignore_tags]):
                pass
            elif self.part_dict[m_name] in testing_tags:
                testing_set.append(j)
            elif self.part_dict[m_name] >= 0:
                training_set.append(j)

        # 3) Distribute entries to training or testing sets accordingly
        training = {}
        testing = {}
        training['inputs'] = np.take(self.all_data, training_set, axis=0)
        testing['inputs'] = np.take(self.all_data, testing_set, axis=0)
        train_samples, in_neurons, feature_length = training['inputs'].shape
        test_samples = len(testing['inputs'])

        training['inputs'] = training['inputs'].transpose([1, 0, 2])
        testing['inputs'] = testing['inputs'].transpose([1, 0, 2])

        training['outputs'] = np.take(self.m_energies, training_set)
        testing['outputs'] = np.take(self.m_energies, testing_set)

        comp_strings = ['-'.join([str(el) for el in com])
                        for com in self.compositions]
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

        training['sizes'] = np.take(self.sizes, training_set)
        testing['sizes'] = np.take(self.sizes, testing_set)

        training['names'] = np.take(self.m_names, training_set, axis=0)
        testing['names'] = np.take(self.m_names, testing_set, axis=0)

        system = {'sys_elements'   : np.string_(self.sys_elements),
                  'max_per_element': self.max_per_element}
        return system, training, testing

    def generate_partitions(self, filename, split_ratio, k=10, simple=True):
        """
        Generate comma-separated value file with training/testing
        designations for each fingerprint. Monte Carlo
        method to solve multi-objective knapsack problem.

        Non-simple: Groups are defined by the second-to-last number
        in each identifier string, and are kept intact during distribution.

        Args:
            filename (str): Filename of .csv (e.g. partitions.csv)
            split_ratio (list): Ratio of training, validation, holdout, etc.
                e.g. [2.0, 1.0, 1.0] to set aside half for testing
                and a quarter each for validation and holdout. Number of
                entries, minus one, equal number of non-training bins.
            k (int): Optional number of subsamples for stratified k-fold.
                e.g. 10 for 10-fold cross validation.
            simple (bool): whether to randomly distribute samples individually.
                (False: preserve groups when partitioning)
        """
        assert self.m_energies.tolist()  # ensure read_fingerprints() completed
        assert k > 1 or split_ratio != [1.0]

        # normalize to sum to 1
        split_ratio = np.divide(split_ratio, np.sum(split_ratio))
        k = max(1, k)
        v = len(split_ratio) - 1
        total_entries = len(self.m_names)
        n_bins = v + k

        # number of subsamples for training and each validation/testing bin
        val_bin_sizes = [(total_entries * fraction)
                         for fraction in split_ratio]
        n_train_samples = val_bin_sizes[0]
        # evenly divide training subsamples into k bins
        train_bin_size = n_train_samples / k
        bin_eq_sizes = val_bin_sizes[1:] + [train_bin_size] * k

        max_iters = self.settings['max_iters']
        size_cost = self.settings['size_cost']
        energy_cost = self.settings['energy_cost']
        bin_energy_cost = self.settings['bin_energy_cost']

        if simple:
            self.m_groups = [name.split('.')[-1] for name in self.m_names]
        else:
            self.m_groups = [name.split('.')[0].split('_')[-2]
                             for name in self.m_names]
        comps = {group: str(comp) for group, comp
                 in zip(self.m_groups, self.compositions)}
        energy_dict = {group: energy for group, energy
                       in zip(self.m_groups, self.m_energies)}
        groups_set = sorted(list(set(self.m_groups)), key=comps.get)
        size_dict = {group: self.m_groups.count(group) for group in groups_set}
        bins_dict = {}

        # 1) Generate subsamples for stratified k-fold cross validation
        bins = [groups_set[i::n_bins] for i in range(n_bins)]
        iteration = 0
        bins_best = bins[:]
        cost_best = np.inf
        e_range = np.max(self.m_energies) - np.min(self.m_energies)
        bins = [sorted(bin_, key=size_dict.get) for bin_ in bins]
        mean_e, var_mean_e = zip(*[mean_energy(bin_,
                                               energy_dict,
                                               size_dict)
                                   for bin_ in bins])
        # 2) Iterate to maximize variation within bins and
        #    minimize variation across bins while maintaining bin size
        try:
            while iteration < max_iters:
                iteration += 1
                # resort each bin's groups from lowest to highest energy
                bins = [sorted(bin_, key=energy_dict.get) for bin_ in bins]
                bin_diffs = np.subtract(mean_e, np.mean(mean_e))
                lowest_bin = int(np.argmin(bin_diffs))
                highest_bin = int(np.argmax(bin_diffs))
                # randomly move a group from the lowest-mean-energy-bin
                # to the highest-mean-energy-bin and vice versa
                rand_ind = int(np.random.rand() * len(bins[lowest_bin]))
                bins[highest_bin].append(bins[lowest_bin].pop(rand_ind))
                rand_ind = int(np.random.rand() * len(bins[highest_bin]))
                bins[lowest_bin].append(bins[highest_bin].pop(rand_ind))
                # resort each bin's groups from least samples to most samples
                bins = [sorted(bin_, key=size_dict.get) for bin_ in bins]
                bin_sizes = [np.sum([size_dict[entry] for entry in bin_])
                             for bin_ in bins]
                bin_diffs = np.subtract(bin_sizes, bin_eq_sizes)
                # randomly move a group from the largest bin to the
                # smallest bin,relative to desired sizes
                smallest_bin = int(np.argmin(bin_diffs))
                largest_bin = int(np.argmax(bin_diffs))
                rand_ind = int(np.random.rand() * len(bins[largest_bin]))
                bins[smallest_bin].append(bins[largest_bin].pop(rand_ind))
                # calculate cost function

                bin_sizes = [np.sum([size_dict[entry] for entry in bin_])
                             for bin_ in bins]
                bin_diffs = np.subtract(bin_sizes, bin_eq_sizes)
                var_bin_diff = np.var(np.abs(bin_diffs))
                mean_e, var_mean_e = zip(*[mean_energy(bin_,
                                                       energy_dict,
                                                       size_dict)
                                           for bin_ in bins])
                var_energies = np.mean(np.divide(var_mean_e, e_range))
                var_bin_energies = np.var(mean_e)

                cost = (size_cost * var_bin_diff
                        + bin_energy_cost * var_bin_energies
                        - energy_cost * var_energies)
                if cost < cost_best:
                    bins_best = list(bins)
                    cost_best = float(cost)
                    print('Best Trial: {}'.format(iteration), end='   \r')

        except (KeyboardInterrupt, SystemExit):
            print('\n')

        best_bin_sizes = [np.sum([size_dict[entry] for entry in bin_])
                          for bin_ in bins_best]
        best_bin_diffs = np.subtract(best_bin_sizes, bin_eq_sizes)
        print('\nFinal bin sizes:', best_bin_sizes)
        print('\nFinal bin deltas:', best_bin_diffs)

        mean_e, var_mean_e = zip(*[mean_energy(bin_,
                                               energy_dict,
                                               size_dict)
                                   for bin_ in bins_best])
        std_e = np.sqrt(var_mean_e)
        for j, bin_ in enumerate(bins_best):
            print('k = {0} mean E: {1:.2f}, std: {2:.2f}'.format(j - v,
                                                                 mean_e[j],
                                                                 std_e[j]))
            for entry in bin_:
                bins_dict[entry] = j - v

        # 3) Assign designation tags to each molecule's fingerprint
        organized_entries = []
        sample_bins = {}
        for m_name in sorted(self.m_names):
            group = self.m_groups[self.m_names.index(m_name)]
            bin_designation = bins_dict[group]
            organized_entries.append([m_name, str(bin_designation)])
            sample_bins[m_name] = bin_designation
        organized_entries = sorted(organized_entries, key=lambda x: x[1])
        # 4) Write to file
        header = 'Name,Designation;\n'
        lines = [','.join(pair) for pair in organized_entries]
        with open(filename, 'w') as fil:
            text = header + ';\n'.join(lines)
            fil.write(text)

        self.part_dict = sample_bins

    def plot_composition_distribution(self, filename=None, even=0):
        assert self.part_dict  # from load_partitions or generate_partitions
        assert self.compositions.tolist()  # from load_preprocessed

        bins = {}
        for m_name, bin_ in sorted(self.part_dict.items()):
            composition = self.compositions[self.m_names.index(m_name)]
            bins.setdefault(bin_, []).append(composition)
        n_bins = len(bins.keys())
        labels = ['k={}'.format(bin_tag) for bin_tag in bins.keys()]
        n_v = len([x for x in bins.keys() if int(x) < 0])
        n_k = n_bins - n_v

        vmark = 'x'
        kmark = 'o'
        holdout = False
        kfold = False
        cols = []
        vcols = []
        kcols = []
        if n_v > 1:
            holdout = True
            vcols = plt.cm.Dark2(np.linspace(0, 1, n_v))
            cols = vcols
            vbins = {key: bins[key] for key in bins.keys() if int(key) < 0}
        if n_k > 1:
            kfold = True
            kcols = plt.cm.rainbow(np.linspace(0, 1, n_k))
            cols = kcols
            kbins = {key: bins[key] for key in bins.keys() if int(key) >= 0}
        if n_k > 1 and n_v > 1:
            cols = np.concatenate([vcols, kcols])

        marks = [vmark] * n_v + [kmark] * n_k

        nnary = len(self.compositions.shape) - self.compositions.shape.count(1)
        print(nnary)
        if nnary == 1:
            # Unary system, plot histogram
            if even > 0:
                # evenly spaced bins
                data = np.stack(bins.values())
                plt.hist(data, even, histtype='bar', colors=cols,
                         labels=labels, stacked=False)
            else:
                # one bar per composition for sparse composition space
                comps = self.compositions.flatten().tolist()
                composition_set = sorted(set(comps))
                bars = np.arange(len(composition_set))
                bottom_data = [0 for comp in composition_set]
                width = 0.35
                for bin_, col in zip(sorted(bins.keys()), cols):
                    print(bin_, col)
                    data_new = [bins[bin_].count(comp)
                                for comp in composition_set]

                    print('bottom:', bottom_data)
                    plt.bar(bars, data_new, width, color=col,
                            bottom=bottom_data, label=bin_)
                    bottom_data = np.add(data_new, bottom_data).tolist()

                plt.ylabel('Samples')
                plt.xticks(bars, [str(comp) for comp in composition_set])
                plt.legend()
                plt.show()

        if nnary == 2:
            tick_spacing = 2

            if kfold and holdout:
                fig, axes = plt.subplots(nrows=1, ncols=3)
                ax0, ax1, ax2 = axes.flatten()

                scatter_compositions(ax1, vbins, '', self.sys_elements,
                                     cols=kcols)
                scatter_compositions(ax2, kbins, '', self.sys_elements,
                                     cols=vcols)
            elif kfold:
                fig, axes = plt.subplots(nrows=1, ncols=2)
                ax0, ax2 = axes.flatten()
                scatter_compositions(ax2, kbins, '', self.sys_elements,
                                     cols=kcols)
            elif holdout:
                fig, axes = plt.subplots(nrows=1, ncols=2)
                ax0, ax1 = axes.flatten()
                scatter_compositions(ax1, vbins, '', self.sys_elements,
                                     cols=vcols)
            else:
                fig, axes = plt.subplots(nrows=1, ncols=1)
                ax0 = axes.flatten()

            x, y = zip(*self.compositions)
            maxes = [max(x) + 1, max(y) + 1]
            h = ax0.hist2d(x, y, bins=maxes,
                           range=[[0, maxes[0]], [0, maxes[1]]],
                           norm=LogNorm())
            ax0.set_title('Overall distribution')
            ax0.set_xlabel(self.sys_elements[0])
            ax0.set_ylabel(self.sys_elements[1])
            ax0.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            ax0.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
            fig.colorbar(h[3], ax=ax0)
            fig.tight_layout()
            plt.show()


def scatter_compositions(ax, bins, title, sys_elements, cols, mark='o'):
    n_bins = len(bins.keys())

    tick_spacing = 2
    xos = np.cos(np.linspace(0, 2 * np.pi, n_bins+1))[:-1] * 0.1
    yos = np.sin(np.linspace(0, 2 * np.pi, n_bins+1)[:-1]) * 0.1
    for (bin_, col, xo, yo) in zip(sorted(bins.keys()), cols, xos, yos):
        x, y = zip(*bins[bin_])
        x = np.add(x, xo)
        y = np.add(y, yo)
        ax.scatter(x, y, s=10, color=col, marker=mark, label=bin_)
    ax.legend()
    ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.xaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing / 2))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    ax.yaxis.set_minor_locator(ticker.MultipleLocator(tick_spacing / 2))
    ax.grid(color='k', which='both')
    ax.set_title(title)
    ax.set_xlabel(sys_elements[0])
    ax.set_ylabel(sys_elements[1])

def mean_energy(bin_, energy_dict, size_dict):
    """
    Weighted average of energy. Intermediate function for
    sorting stratified kfold chunks.
    """
    group_energies, group_sizes = zip(*[(energy_dict[entry], size_dict[entry])
                                        for entry in bin_])
    sample_energies = np.repeat(group_energies, group_sizes)
    mean_e = np.mean(sample_energies)
    var_mean_e = np.var(sample_energies)
    return mean_e, var_mean_e


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
        st_min = np.max(fp_as_grid, axis=0)                                    #min#
        st_max = np.min(fp_as_grid, axis=0)                                    #max#
        st_mean = np.mean(fp_as_grid, axis=0)
        standards.append(np.stack([st_min, st_max, st_mean], axis=0))
        #normalized_grid = skp.normalize(fp_as_grid, axis=0, norm='max')
        normalized_grid = (fp_as_grid - st_min) / (st_max - st_min)            #normalize equation#
        for_calculator_G1 = [standards[0][0] , standards [0][1]]
        for_calculator_G2 = [st_min , st_max]
        #for_calculator_min = [standards[0][0] , st_min]
        #for_calculator_max = [standards[0][1] , st_max]
        #for_calculator_G = [for_calculator_min, for_calculator_max]

        #np.savetxt('max_min_value_G.csv',for_calculator_G,delimiter=',')
        np.savetxt('max_min_value_G2.csv',for_calculator_G2,delimiter=',')
        np.savetxt('max_min_value_G1.csv',for_calculator_G1,delimiter=',')          

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


