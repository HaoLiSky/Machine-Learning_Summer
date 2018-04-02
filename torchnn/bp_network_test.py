"""
Network for training/testing using Keras.
"""
import os
import sys
import h5py
import numpy as np
import sklearn.preprocessing as skp
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
# from torch.utils.data import Dataset, DataLoader
from torch import LongTensor, FloatTensor
from time import time
from itertools import combinations_with_replacement
import matplotlib.pyplot as plt


# arbitrary fixed seed for reproducibility
np.random.seed(8)
torch.manual_seed(8)


# class Tensorset(Dataset):
#     def __init(self, dset):
#         self.inputs = dset['input']
#         self.outputs = dset['output']
#
#     def __len__(self):
#         return len(self.inputs)
#
#     def __getitem__(self, idx):
#         sample = {}
#         for k, v in self.inputs[idx].items():
#             sample[k] = torch.from_numpy(v).pin_memory()
#         energy = self.outputs[idx]
#         return sample, energy


def parse_architecture(setting_string):
    """
    e.x. '112-16-1'  ->  [(0, 112, 16), (1, 16, 1)]

    Args:
        setting_string: Dash-separated layer widths.

    Returns:
        Enumerated list of layer architectures.
    """
    widths = [int(width) for width in setting_string.split('-')]
    inds = range(len(widths)-1)
    inputs = widths[:-1]
    outputs = widths[1:]
    structure = list(zip(inds, inputs, outputs))
    return structure


class AtomNet(nn.Module):
    """
    Atom-level Neural Network
    """
    def __init__(self, structure, tag, d):
        super(AtomNet, self).__init__()
        activation = nn.Softplus()
        do = nn.Dropout(p=d)
        self.layers = []
        self.structure = structure
        self.forces = False
        for i, input_width, layer_width in self.structure:
            # construct subnetwork
            # e.g. "5-5-1" -> 2x5 hidden layers, 1x1 output layer
            # layer = nn.Linear(input_width, layer_width, bias=False)
            layer = nn.Linear(input_width, layer_width)
            self.layers.append(layer)
            self.add_module("{}.{}".format(tag, i), layer)
            if i < len(self.structure) - 1:
                self.layers.append(activation)
                self.add_module("{}.{}".format('softplus', i), activation)
                self.layers.append(do)
                self.add_module("{}.{}".format('dropout', i), do)

    def forward(self, atomic_data):
        if self.forces:
            #atomic_data = atomic_data.type(dtype)
            #fingerprint =
            #h = fingerprint
            h = atomic_data
            for layer in self.layers:
                h = layer(h)
            output = h
            output.backward()
            gradient = atomic_data.grad
            return output, gradient
        else:
            h = atomic_data
            for layer in self.layers:
                h = layer(h)
            output = h
            return output


class MoleculeNet(nn.Module):
    """
    Molecule-level neural network.
    """
    def __init__(self, sys_elements, structure, settings, states=None):
        if not states:
            states = {}
        super(MoleculeNet, self).__init__()
        self.subnet = {}
        # TODO: initialize atomnets with selected loss function from settings
        for el in sys_elements:
            sub = AtomNet(structure, el, settings['dropout'])
            self.subnet[el] = sub
            if el in states:
                self.subnet[el].load_state_dict(states[el], strict=False)
            self.add_module("{}_module".format(el), sub)
            print(sub)

    def forward(self, batch_data):
        batch_output = []
        for mol_data in batch_data:
            at_energies = []
            for el, el_data in mol_data.items():
                el_energies = []
                for at_data in el_data:
                    at_energy = self.subnet[el](at_data)
                    el_energies.append(at_energy)
                at_energies.extend(el_energies)
            mol_energy = torch.cat(at_energies, dim=0).mean(0)
            batch_output.append(mol_energy)
        return torch.cat(batch_output, dim=0)

    def save_all_state_dicts(self, filename):
        states = {}
        for el, subnet in self.subnet.items():
            states[el] = subnet.state_dict()
        torch.save(states, filename)

    def load_all_state_dicts(self, filename):
        print('loaded weights from {}'.format(filename))
        self.load_state_dict(torch.load(filename))

class NetworkHandler:
    """
    Artificial Neural Network.
    """

    def __init__(self, settings, **kwargs):
        self.settings = settings
        self.settings.update(kwargs)
        self.cuda = self.settings['cuda']
        self.test_step = self.settings['test_step']
        self.test_batches = self.settings['test_batches']
        self.loss = nn.MSELoss()
        self.batch_loss = nn.MSELoss(reduce=False)
        if self.cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = FloatTensor

    def load_data(self, filename, partition_file):
        """
        Args:
            filename (str): File with preprocessed fingerprint data.
                e.g. fingerprints.h5
            partition_file (str): File with partition designations for each
                fingerprint identifier.
            run_settings (dict): Run-specific settings, i.e. including
                a unique value of k_test for k-fold cross validation.
        """
        libver = self.settings['libver']
        index = self.settings['index']
        part = PartitionProcessor()
        part.load_preprocessed(filename, libver=libver, index=index)
        part.load_partitions_from_file(partition_file)
        data = part.get_network_inputs(testing_tags=self.settings['test_tags'],
                                       dtype=self.dtype, GPU=self.cuda)
        system, np_train, np_test = data

        self.sys_elements = [val.decode('utf-8')
                             for val in system['sys_elements']]

        self.input_train = np_train['inputs']
        self.input_test = np_test['inputs']
        self.target_train = np_train['outputs']
        self.target_test = np_test['outputs']

        self.testing_mean = torch.mean(self.target_test)
        self.testing_max = torch.max(self.target_test)
        self.testing_min = torch.min(self.target_test)
        self.test_batch_inds = np.array_split(np.arange(len(self.target_test)),
                                              self.test_batches)
        self.n_batches = np.round(len(self.target_train)
                                  / float(self.settings['batch_size']))

    def check_integrity(self):
        """
        Ensure model does not return infinity or NaN values.
        If so, there is, most likely, a problem with the input data.
        """
        self.model.eval()
        starts, ends = get_intervals(len(self.input_train),
                                         self.settings['batch_size'])
        for start, end in zip(starts, ends):
            untrained_sample = self.model(self.input_train[start: end]).data
            inf = torch.from_numpy(np.asarray([np.inf])).type(self.dtype)
            assert (untrained_sample == untrained_sample).all()
            assert (untrained_sample != inf).all()

    def train(self):
        np.random.shuffle(self.all_inds)
        self.model.train()  # Turn on training mode, incl. dropout
        epoch_loss = 0
        batch_steps = 0
        for batch_inds in np.array_split(self.all_inds, self.n_batches):
            # Loop through batches
            self.optimizer.zero_grad()
            output = self.model([self.input_train[ind] for ind in batch_inds])
            batch_target = self.dtype(
                    [self.target_train[ind] for ind in batch_inds])
            target = Variable(batch_target).type(self.dtype)
            loss = self.loss(output, target)
            batch_loss = loss.data.cpu().numpy()[0]
            epoch_loss += batch_loss
            loss.backward()
            self.optimizer.step()
            batch_steps += 1
        epoch_loss /= batch_steps
        return epoch_loss

    def test(self):
        self.model.eval()  # Evaluation mode turns off dropout
        # Make weights checkpoint
        torch.save(self.model.state_dict(), self.settings['checkpoint_fn'])
        batch_outputs = []
        sample_losses = []
        for inds in self.test_batch_inds:
            # Loop over batches
            out = self.model([self.input_test[ind] for ind in inds])
            batch_target = self.dtype([self.target_test[ind]
                                       for ind in inds])
            var = Variable(batch_target,
                           requires_grad=False).type(self.dtype)
            batch_loss = self.batch_loss(out, var).data.cpu()
            sample_losses.extend(batch_loss.numpy().tolist())
            batch_outputs.extend(out.data.cpu().numpy().tolist())
        test_loss = np.mean(sample_losses)
        test_max = np.max(batch_outputs)
        test_min = np.min(batch_outputs)
        test_range = '{0:.1f},{1:.1f}'.format(test_max, test_min)
        test_mean = np.mean(batch_outputs)
        test_stats = {'range': test_range, 'mean': test_mean,
                      'outputs': batch_outputs}
        return test_loss, test_stats

    def fit(self, **kwargs):
        """

        """
        settings = dict(self.settings)
        settings.update(kwargs)  # for per-run hyperparameters
        self.architecture = parse_architecture(settings['structure'])
        self.layers_id = 0
        # TODO: regularization, optimizer, activation from settings
        self.model = MoleculeNet(self.sys_elements, self.architecture,
                                 settings)
        # Move model and weights to GPU
        if settings['cuda']:
            self.model.cuda()
            torch.cuda.synchronize()
        self.model.float()
        # Load weights from file
        if os.path.isfile(settings['weights_fn']):
            try:
                self.model.load_all_state_dicts(settings['weights_fn'])
            except RuntimeError:
                print('Invalid Weights file')
        # Check Integrity
        if self.settings['check_nan']:
            self.check_integrity()
        # Enumerate training set
        self.all_inds = np.arange(len(self.target_train))
        # TODO: select optimizer and activations from settings
        self.optimizer = optim.Adam(self.model.parameters())
        train_losses = []
        test_losses = []
        t0 = time()
        print('starting training')  # TODO: print header
        try:
            for epoch in range(settings['epochs']):
                train_loss = self.train()
                print(train_loss)
                train_losses.append(train_loss)

                if epoch % self.test_step == 0:
                    test_loss, test_stats = self.test()
                    dm = self.testing_mean - test_stats['mean']
                    print('Difference in Means:', '{0:.1f}'.format(dm))
                    print('Test  :', test_stats['range'])
                    print('Target:', '{0:.1f},{1:.1f}'.format(self.testing_max,
                                                              self.testing_min))
                    plt.close('all')
                    fig, ax = plt.subplots(1, 1)
                    ax.scatter(self.target_test, test_stats['outputs'], c='r')
                    ax.plot([-1800, -100], [-1800, 100], c='k')
                    ax.axis('equal')
                    ax.set_aspect('equal', 'box')
                    ax.yaxis.set_label_text('NNP')
                    ax.xaxis.set_label_text('DFT')
                    ti = 'Epoch {0}, Train={1:.1f}, Test={2:.1f}'
                    ax.set_title(ti.format(epoch, train_loss, test_loss))
                    fig.savefig('{}.png'.format(epoch), dpi=300)
                test_losses.append(test_loss)

                print(str(epoch).ljust(5),
                      '| {0:.2f} | {1:.2f}'.format(train_loss, test_loss),
                      '| {0:.0f} sec'.format(time() - t0))
                sys.stdout.flush()
        except (KeyboardInterrupt, SystemExit):
            print('\n')
        # 8) record weights and history of losses
        torch.save(self.model.state_dict(), settings['weights_fn'])
        a = np.pad(train_losses, (0, max(0, len(test_losses)
                                         - len(train_losses))), 'constant',
                                  constant_values=0)
        b = np.pad(test_losses, (0, max(0, len(train_losses)
                                         - len(test_losses))), 'constant',
                                  constant_values=0)
        np.savetxt('asdf.hist',
                   np.vstack((a, b)).transpose(),
                   delimiter=',', newline=';\n')

        return test_losses[-1]


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
        self.m_attrs = []
        self.m_energies = []
        self.m_element_counts = []
        self.standards = []

        libver = self.settings['libver']
        descriptor = self.settings['descriptor']
        index = self.settings['index']
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
                               for key in sorted(fp.dsets_dict.keys())
                               if key.find('G_') == 0])
                self.dg.append([fp.dsets_dict[key]
                                for key in sorted(fp.dsets_dict.keys())
                                if key.find('dG_') == 0])
                energy = float(fp.energy_val) * 1000
                # energy per atom
                self.m_energies.append(energy)
                self.m_element_counts.append(fp.element_counts)

                self.m_attrs.append({'energy': energy,
                                     'composition': fp.element_counts,
                                     'size': fp.natoms})

                if not set(fp.elements_set).issubset(set(self.sys_elements)):
                    continue  # skip if not part of system
            self.g = [list(g) for g in zip(*self.g)]
            self.dg = [list(dg) for dg in zip(*self.dg)]
            print('Read {} fingerprints from {}'.format(len(self.m_energies),
                                                        filename))

    def preprocess_fingerprints(self):
        """
        Preprocess fingerprints.
        """
        assert self.g  # ensure read_fingerprints() completed
        # 2) get interactions per fingerprint
        pair_i = list(combinations_with_replacement(self.sys_elements, 1))
        triplet_i = list(combinations_with_replacement(self.sys_elements, 2))
        alpha = [len(pair_i), len(triplet_i)]
        # 3) normalize and regenerate by element
        normalized, self.standards = normalize_to_vectors(self.g, alpha)
        regenerated = split_by_element(normalized, self.m_element_counts,
                                       self.sys_elements)
        self.all = regenerated

    def to_file(self, filename):
        """
        Write preprocessed fingerprints to file.

        Args:
            filename (str): Output file (e.g. preprocessed.h5).
        """
        assert self.standards  # ensure preprocess_fingerprints() completed
        libver = self.settings['libver']
        with h5py.File(filename, 'w', libver=libver) as h5f:
            for m_name, attrs, m_dict in zip(
                    self.m_names, self.m_attrs, self.all):
                path = 'Preprocessed/{}'.format(m_name)
                write_to_group(h5f, path, attrs, m_dict)

            scaling_standards = {'standard_{}'.format(j): standard
                                 for j, standard in enumerate(self.standards)}
            write_to_group(h5f, 'system',
                           {'sys_elements': np.string_(self.sys_elements)},
                           scaling_standards)


class PartitionProcessor:
    def __init__(self):
        self.all_data = []
        self.sizes = []
        self.compositions = []
        self.m_energies = []

    def load_preprocessed(self, filename, libver='latest', index=':'):
        """
        Args:
            filename (str): File with preprocessed fingerprint data.
                e.g. preprocessed.h5
            libver (str): Optional h5py argument for i/o. Defaults to 'latest'.
        """
        # 1) Read data from hdf5 file
        with h5py.File(filename, 'r', libver=libver) as h5f:
            self.sys_elements = [specie.decode('utf-8')
                                 for specie in
                                 h5f['system'].attrs['sys_elements']]

            self.m_names = list(h5f.require_group('Preprocessed').keys())
            self.m_names = np.asarray(sorted(self.m_names))
            self.m_names = self.m_names[slice_from_str(index)]
            for j, m_name in enumerate(self.m_names):
                print('read', j, end='\r')
                path = 'Preprocessed/{}'.format(m_name)

                attrs_dict, m_dict = read_from_group(h5f, path)
                m_size = attrs_dict['size']
                m_energy = attrs_dict['energy']
                m_comp = attrs_dict['composition']
                m_dict = dict((k, v) for k, v in m_dict.items() if v.any())
                self.all_data.append(m_dict)
                self.sizes.append(m_size)
                self.m_energies.append(m_energy)
                self.compositions.append(m_comp)

        if os.path.isfile('ignore_tags'):
            with open('ignore_tags', 'r') as file_:
                ignore_tags = file_.read().split('\n')
                self.ignore_tags = list(filter(None, ignore_tags))
        else:
            self.ignore_tags = []

    def load_partitions_from_file(self, filename):
        """
        Class wrapper for read_partitions().
        """
        self.part_dict = read_partitions(filename)

    def get_network_inputs(self, testing_tags=(), verbosity=1, dtype=FloatTensor, GPU=False):
        """
         Args:
             testing_tags (int): Designation tags for testing.
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
        train_samples = len(training['inputs'])
        test_samples = len(testing['inputs'])

        train_energies = np.take(self.m_energies, training_set)
        test_energies = np.take(self.m_energies, testing_set)

        for mol_dict in training['inputs']:
            for k, v in mol_dict.items():
                mol_dict[k] = Variable(torch.from_numpy(np.expand_dims(v, 1)),
                                       requires_grad=False).type(dtype)
                if GPU:
                    mol_dict[k].cuda()
                print(mol_dict[k].shape, end='\r')

        for mol_dict in testing['inputs']:
            for k, v in mol_dict.items():
                mol_dict[k] = Variable(torch.from_numpy(np.expand_dims(v, 1)),
                                       requires_grad=False).type(dtype)
                if GPU:
                    mol_dict[k].cuda()
                print(mol_dict[k].shape, end='\r')

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
            print('Unique compositions in training set:')
            print(train_u)
            print('Unique compositions in testing set:')
            print(test_u)
            print('Shared compositions in both training and testing sets:')
            print(common)

        training['sizes'] = np.take(self.sizes, training_set).astype(float)
        testing['sizes'] = np.take(self.sizes, testing_set).astype(float)

        training['outputs'] = torch.from_numpy(train_energies
                                               / training['sizes'])
        testing['outputs'] = torch.from_numpy(test_energies/testing['sizes'])
        if GPU:
            training['outputs'].cuda()
            testing['outputs'].cuda()

        training['names'] = np.take(self.m_names, training_set, axis=0)
        testing['names'] = np.take(self.m_names, testing_set, axis=0)

        system = {'sys_elements': np.string_(self.sys_elements)}
        return system, training, testing


class Fingerprint:
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
                   dset_types=None, **kwargs):
    """
    Writes datasets and attributes to specified group.

    Args:
        h5f: h5py file.
        group_path (str): Path to group
        dict_dsets (dict): Dataset names and ndarrays.
        dict_attrs (dict): Attribute names and values.
        dset_types (dict): Optional data types for dataset(s).
    """
    if not dset_types:
        dset_types = {}
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


def read_partitions(filename):
    """
    Args:
        filename: File with training/testing partitions (e.g. partitions.csv)

    Returns:
        Dictionary of molecule ids and their partition designations.
    """
    with open(filename, 'r') as fil:
        lines = fil.readlines()[1:]  # skip header
    lines = [line.replace('\n', '').replace(';', '') for line in lines]
    entries = [line.split(',') for line in lines if line]
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
        st_min = np.max(fp_as_grid, axis=0)
        st_max = np.min(fp_as_grid, axis=0)
        st_mean = np.mean(fp_as_grid, axis=0)
        standards.append(np.stack([st_min, st_max, st_mean], axis=0))
        #normalized_grid = skp.normalize(fp_as_grid, axis=0, norm='max')
        normalized_grid = skp.robust_scale(fp_as_grid)

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


def split_by_element(input_data, compositions, sys_elements):
    """
    Slice fingerprint arrays by elemental composition

    Args:
        input_data: 2D array of structures/features.
            i.e. output of normalize_inputs()
        compositions: list of "layer" heights (per element) in # of atoms.

    Returns:
        output_data: List of dictionaries per molecule.
    """
    all_data = []
    for molecule, initial_layers in zip(input_data, compositions):
        data = np.asarray(molecule)
        slice_pos = np.cumsum(initial_layers)[:-1]
        # row indices to slice to create n sections
        data_sliced = np.split(data, slice_pos)
        molecule_data = {k: v for k, v in zip(sys_elements, data_sliced)}
        all_data.append(molecule_data)
    return all_data


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


def grid_string(entries, max_length=79, separator=', ', newline=';\n'):
    if len(entries) == 0:
        return ''
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
    grid_lines += grid_line + newline
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


def get_intervals(n_samples, batch_size):
    starts = np.arange(n_samples / batch_size)
    starts = (starts * batch_size).astype(int)
    ends = starts[1:]
    np.append(ends, n_samples + 1)
    return starts, ends


if __name__ == '__main__':
    try:
        test_tags = [int(sys.argv[1])]

        if sys.argv[2].lower() == 'gpu':
            GPU = True
            print('GPU on')
            dtype = torch.cuda.FloatTensor
        else:
            GPU = False

    except:
        test_tags = [0]
        GPU = False
        dtype = FloatTensor

    # Create and train Keras models
    inputs_name = 'preprocessed_robust.hdf5'
    partitions_file = 'partitions.csv'
    settings = {'libver': 'latest',
                'verbosity': 1,
                'record': 1,
                'cuda': GPU,
                'check_nan': False,
                'test_step': 20,
                'test_batches': 5,
                'weights_fn': 'weights.dict',
                'checkpoint_fn': 'checkpoint.dict',
                'test_tags': test_tags, 'dropout': 0.15,
                'structure':  '112-64-64-1',
                'epochs': 10000, 'batch_size': 1024,
                'index': '::3'}

    network = NetworkHandler(settings)

    network.load_data(inputs_name, partitions_file)
    network.fit()

    # for k in [0, 1, 2]:
    #     run_settings = {'test_chunk': [k], 'dropout': 0.15,
    #                     'structure': '64-64-1',
    #                     'epochs': 1000, 'batch_size': 256}
    #     network.load_data(inputs_name, partitions_file,
    #                       run_settings=run_settings)
    #     final_loss = network.fit(run_settings)
    #     print('\n\nFinal loss:', final_loss)
