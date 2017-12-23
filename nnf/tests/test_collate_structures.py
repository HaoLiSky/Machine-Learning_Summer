"""
Test structure parsing into .hdf5 and fidelity of data, within precision of
1e-4. Test case uses 10 structures with 1~60 atoms in a
5x5x5 angstrom cubic cell, distributed randomly in [1,4)
along each axis to ignore periodic boundary errors.
"""
import unittest
import os
import h5py
import numpy as np
from io_structures import collate_structures
from framework_cli import validate_hdf5
from test_utils import generate_random_cif


class StructuresParseTestCase(unittest.TestCase):
    def setUp(self):
        """

        Prepares cif to parse.

        """
        self.n = 100
        self.sys_elements = ['Ba', 'Ti', 'O']
        self.max_per_element = [4, 5, 6]
        self.cell_length = 5

        self.tempfilename = 'test_{}.cif'.format(''.join(self.sys_elements))
        generate_random_cif(self.tempfilename, self.sys_elements,
                            self.max_per_element, self.cell_length, self.n)

        self.n_elements = len(self.sys_elements)
        self.property_data = np.random.rand(self.n, )
        self.property_data_pass = [str(x) for x in self.property_data]
        self.prop_test = True

    def tearDown(self):
        """
        Clear temp files.
        """
        os.remove(self.tempfilename)
        os.remove('structure.test.hdf5')

    def test_structures_to_hdf5(self):
        collate_structures(self.tempfilename, 'structure.test.hdf5',
                           self.sys_elements, self.property_data_pass,
                           form='cif')
        num_collated = validate_hdf5('structure.test.hdf5')[0]
        self.validation_test = (num_collated == self.n)

        with h5py.File('structure.test.hdf5', 'r', libver='latest') as h5f:
            s_names = [name.decode('utf-8')
                       for name in h5f['system']['s_names_list'][()]]
            structures_dset = h5f['structures']
            compositions = []
            for j, s_name in enumerate(s_names):
                dset = structures_dset[s_name]
                compositions.append(dset.attrs['element_counts'])

                energy_search = np.isclose(self.property_data[:, None],
                                           float(dset.attrs['energy']),
                                           atol=1e-4).all(-1)

                if not np.any(energy_search):
                    self.prop_test = False
            max_observed = np.amax(compositions, axis=0)
            self.max_per_element_test = np.all(np.isclose(max_observed,
                                                          self.max_per_element,
                                                          atol=2))

        self.s_tests = [self.max_per_element_test,
                        self.validation_test,
                        self.prop_test]
        self.assertTrue(np.all(self.s_tests))


if __name__ == '__main__':
    unittest.main()
