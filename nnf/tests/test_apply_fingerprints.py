"""
Test fingerprint creation and writing into .hdf5, and fidelity of data,
within precision of 1e-4. Test case uses 10 structures with 1~60 atoms in a
5x5x5 angstrom cubic cell, distributed randomly in [1,4)
along each axis to ignore periodic boundary errors.
Placeholder descriptor produces random values for pairwise and triplet
interactions, mimicking Parrinello-Behler outputs.
Parameters used: 5 per set; 4 pairwise sets, 6 triplet sets.
"""
import unittest
import os
import h5py
import numpy as np
from io_structures import collate_structures
from io_fingerprints import apply_descriptors
from framework_cli import validate_hdf5
from test_utils import generate_random_cif


class ApplyFingerprintsTestCase(unittest.TestCase):
    def setUp(self):
        """
        Generate temp .cif and intermediate .hdf5 file of structures.
        """
        self.descriptor = 'bp'

        self.n = 10
        self.sys_elements = ['Ba', 'Ti', 'O']
        self.max_per_element = [2, 4, 6]
        self.cell_length = 5

        self.tempfilename = 'test_{}.cif'.format(''.join(self.sys_elements))
        self.natoms_list = generate_random_cif(self.tempfilename,
                                               self.sys_elements,
                                               self.max_per_element,
                                               self.cell_length, self.n)

        self.property_data = np.random.rand(self.n, )
        self.property_data_pass = [str(x) for x in self.property_data]
        self.prop_test = True

        collate_structures(self.tempfilename, 'structure.test.hdf5',
                           self.sys_elements, self.property_data_pass,
                           form='cif')
        validate_hdf5('structure.test.hdf5')
        self.dim_data = np.asarray([[natoms, 3, 1, 6, 1]
                                    for natoms in self.natoms_list])
        self.prop_test = True
        self.dim_test = True
        self.parameters = (np.asarray([(6, 1, 1, 1, 1)]),
                           np.asarray([(6, 1, 1, 1, 1)]))

    def tearDown(self):
        """
        Remove temp files.
        """
        os.remove(self.tempfilename)
        os.remove('structure.test.hdf5')
        os.remove('fingerprint.test.hdf5')

    def test_structures_to_hdf5(self):
        apply_descriptors('structure.test.hdf5', 'fingerprint.test.hdf5',
                          self.sys_elements, self.parameters,
                          descriptor=self.descriptor, primes=True)
        num_fingerprints = validate_hdf5('fingerprint.test.hdf5')[1]
        self.validation_test = (num_fingerprints == self.n)
        with h5py.File('fingerprint.test.hdf5', 'r', libver='latest') as h5f:
            s_names = sorted(list(h5f['structures'].keys()),
                             key=lambda x: int(x.split('_')[-1]))
            self.n_structures_test = len(s_names) == self.n
            structures_dset = h5f['structures']
            for j, s_name in enumerate(s_names):
                dset = structures_dset[s_name]
                g_1, g_2 = [dset[x][()] for x in list(dset.keys())][:2]
                dset_dims = [g_1.shape[0], g_1.shape[1], g_1.shape[2],
                             g_2.shape[1], g_2.shape[2]]
                dim_search = np.isclose(self.dim_data[:, None],
                                        dset_dims,
                                        atol=0).all(-1)
                if not np.any(dim_search):
                    self.dim_test = False

                energy_search = np.isclose(self.property_data[:, None],
                                           float(dset.attrs['energy']),
                                           atol=1e-4).all(-1)
                if not np.any(energy_search):
                    self.prop_test = False

        self.s_tests = [self.validation_test,
                        self.dim_test,
                        self.prop_test]
        self.assertTrue(np.all(self.s_tests))


if __name__ == '__main__':
    unittest.main()
