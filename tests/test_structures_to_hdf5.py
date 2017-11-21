"""
Test structure parsing into .hdf5 and fidelity of data, within precision of
1e-4. Test case uses 10 structures with 1~60 atoms in a
5x5x5 angstrom cubic cell, distributed randomly in [1,4)
along each axis to ignore periodic boundary errors.
"""
import unittest
import os
import ase
import ase.io as ase_io
import h5py
import numpy as np
from representation import structures_to_hdf5


class StructuresParseTestCase(unittest.TestCase):
    def setUp(self):
        """
        Generate temporary .cif file for parsing.
        """
        self.n = 10
        self.system_symbols = ['Ba', 'Ti', 'O']
        self.n_elements = 3
        self.max_per_element = [10, 20, 30]
        self.cell_length = 5
        self.property_data = np.random.rand(self.n, )
        self.comp_coeff = np.zeros((self.n, self.n_elements), dtype='i4')
        for s, nmax in enumerate(self.max_per_element):
            while True:
                self.comp_coeff[:, s] = np.random.randint(nmax + 1,
                                                          size=(self.n,))
                if np.sum(self.comp_coeff[:, s]) > 0:
                    break
                    # random number of atoms of each species per structure
        self.natoms_list = np.sum(self.comp_coeff, axis=1)
        self.coords_list = []
        for natoms in self.natoms_list:
            coords = np.zeros(shape=(natoms, 3))
            for atom in range(natoms):
                while True:
                    proposed = np.random.rand(1, 3)
                    if not np.any([np.allclose(proposed, x, atol=0.01)
                                   for x in coords]):
                        coords[atom, :] = proposed
                        break
            self.coords_list.append(np.add(np.multiply(coords,
                                                       self.cell_length - 2),
                                           1))
        self.species_list_list = []
        self.symbol_set_list = []
        for species_counts in self.comp_coeff:
            species_list = []
            for symbol, ccount in zip(self.system_symbols, species_counts):
                species_list += [symbol] * ccount
            self.species_list_list.append(species_list)
            self.symbol_set_list.append(list(set(species_list)))

        self.unit = np.asarray([[self.cell_length, 0, 0],
                                [0, self.cell_length, 0],
                                [0, 0, self.cell_length]])
        self.periodic = True

        self.structure_list = [ase.Atoms(''.join(species_list),
                                         positions=coords,
                                         cell=self.unit,
                                         pbc=self.periodic)
                               for species_list, coords
                               in
                               zip(self.species_list_list, self.coords_list)]
        ase_io.write('test.cif', self.structure_list, format='cif')
        self.n_structures_test = True
        self.natoms_test = True
        self.symbols_test = True
        self.coords_test = True
        self.num_elem_test = True
        self.prop_test = True
        self.cell_test = True

    def tearDown(self):
        """
        Clear temp files.
        """
        os.remove('test.cif')
        os.remove('structure.test.hdf5')

    def test_structures_to_hdf5(self):
        structures_to_hdf5('test.cif', 'structure.test.hdf5',
                           self.system_symbols, self.property_data, form='cif')
        with h5py.File('structure.test.hdf5', 'r', libver='latest') as h5f:
            self.s_names = sorted(list(h5f['structures'].keys()),
                                  key=lambda x: int(x.split('_')[-1]))
            self.n_structures_test = len(self.s_names) == self.n
            structures_dset = h5f['structures']
            for j, s_name in enumerate(self.s_names):
                dset = structures_dset[s_name]['coordinates']
                if not self.natoms_list[j] == dset.attrs['natoms']:
                    self.natoms_test = False
                if not self.property_data[j] == dset.attrs['property']:
                    self.prop_test = False
                if not self.symbol_set_list[j] == [x.decode('utf-8')
                                                   for x in
                                                   dset.attrs['symbol_set']]:
                    self.symbols_test = False
                if not np.allclose(self.coords_list[j], dset[()], atol=1e-4):
                    self.coords_test = False
                if not np.allclose(self.comp_coeff[j, :],
                                   dset.attrs['species_counts']):
                    self.num_elem_test = False
                if not np.allclose(self.unit, dset.attrs['unit']):
                    self.cell_test = False

        self.s_tests = [self.n_structures_test,
                        self.natoms_test,
                        self.symbols_test,
                        self.coords_test,
                        self.num_elem_test,
                        self.prop_test,
                        self.cell_test]
        self.assertTrue(np.all(self.s_tests))


if __name__ == '__main__':
    unittest.main()
