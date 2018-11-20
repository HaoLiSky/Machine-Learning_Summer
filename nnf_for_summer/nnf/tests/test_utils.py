"""
Utility functions for module testing.
"""
import ase
import ase.io as ase_io
import numpy as np


def generate_random_cif(filename, system_elements, max_per_element,
                        cell_length, n):
    """
    Generate temporary .cif file for parsing.
    """
    n_elements = len(system_elements)
    comp_coeff = np.zeros((n, n_elements), dtype='i4')
    for s, nmax in enumerate(max_per_element):
        while True:
            comp_coeff[:, s] = np.random.randint(nmax + 1,
                                                 size=(n,))
            if np.sum(comp_coeff[:, s]) > 0:
                break
                # random number of atoms of each species per structure
    for i in range(n):
        if np.sum(comp_coeff[i, :]) == 0:
            comp_coeff[i, :] = np.arange(n_elements)
    natoms_list = np.sum(comp_coeff, axis=1)
    coords_list = []
    for natoms in natoms_list:
        coords = np.zeros(shape=(natoms, 3))
        for atom in range(natoms):
            while True:
                proposed = np.random.rand(1, 3)
                if not np.any([np.allclose(proposed, x, atol=0.01)
                               for x in coords]):
                    coords[atom, :] = proposed
                    break
        coords_list.append(np.add(np.multiply(coords,
                                              cell_length - 2),
                                  1))
    species_list_list = []
    symbol_set_list = []
    for species_counts in comp_coeff:
        species_list = []
        for symbol, ccount in zip(system_elements, species_counts):
            species_list += [symbol] * ccount
        species_list_list.append(species_list)
        symbol_set_list.append(list(set(species_list)))

    unit = np.asarray([[cell_length, 0, 0],
                       [0, cell_length, 0],
                       [0, 0, cell_length]])
    periodic = True

    structure_list = [ase.Atoms(''.join(species_list),
                                positions=coords,
                                cell=unit,
                                pbc=periodic)
                      for species_list, coords
                      in
                      zip(species_list_list, coords_list)]
    ase_io.write(filename, structure_list, format='cif')
    return natoms_list
