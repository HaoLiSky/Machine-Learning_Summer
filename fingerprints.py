"""
Fingerprint functions.
"""
import numpy as np
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
from pymatgen.core.structure import Structure, IStructure


def bp_fingerprint(s_data, parameters, system_symbols):
    """

    Parrinello-Behler representation. Computes fingerprints for
    pairwise (g_1) and triple (g_2) interactions.

    Fingerprints are ndarrays of size (n x s x k)
    where n = number of atoms, k = number of parameters given for the
    fingerprint, and s = combinations with replacement of the system's species
    set. When the input's species set is less than the system's species set,
    fingerprints are padded.

    Args:
        s_data: List of data (output of read_structure).
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, symbol_set, species_counts,
     species_list, unit, periodic, property_value) = s_data
    if not set(symbol_set).issubset(set(system_symbols)):
        raise AssertionError(str('-'.join(set(symbol_set))),
                             'not valid for',
                             str('-'.join(set(system_symbols))))

    if not len(species_list) == len(coords):
        print(species_list, coords.shape)
    # assert len(species_list) == len(coords)
    # assert set(symbol_set).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters

    if periodic:
        # to do: replace pymatgen dependency with ase.build
        structure = Structure(unit, species_list, coords,
                              coords_are_cartesian=True)
        unitcell = structure.copy()

        # generate supercell to include all neighbors of atoms in unitcell
        supercell, indices_unitcell = build_supercell(unitcell, R_c=6.0)

        g_1, g_2 = represent_BP(np.asarray(supercell.cart_coords),
                                supercell.atomic_numbers,
                                [para_pairs, para_triplets],
                                periodic=True,
                                indices_unitcell=indices_unitcell)
    else:
        g_1, g_2 = represent_BP(coords, species_list,
                                [para_pairs, para_triplets])

    data = pad_fingerprints([g_1, g_2], symbol_set, system_symbols, [1, 2])
    labels = ['BP_g1', 'BP_g2']
    fingerprints = zip(labels, data)
    return fingerprints


def represent_BP(coords, elements, parameters, periodic=False,
                 indices_unitcell=None):
    """
    Computes the Behler-Parrinello atom-based descriptors
    for each atom in a given structure.

    Args:
        coords: list of [xyz] coords (in Angstroms)
        elements: list of element name strings
        parameters: list of parameters for Behler-Parrinello symmetry functions
            r_cut: cutoff radius (angstroms); default = 6.0
            r_s: pairwise distance offset (angstroms); default = 1.0
            eta: exponent term dampener; default = 1.0
            lambda_: angular expansion switch +1 or -1; default = 1.0
            zeta: angular expansion degree; default = 1.0
        periodic: boolean (False = cluster/molecule,
            True = 3D periodic structure)
        indices_unitcell: list of indices of atoms in the unitcell
                          (only applicable for periodic structures)

    Returns:
        (# atoms, # unique element types, # descriptors) ndarray of G^1s and
        (# atoms, # unique element type pairs, # descriptors) ndarray of G^2s.

    Notes:
        Behler-Parrinello symmetry functions as described in:
        Behler, J; Parrinello, M. Generalized Neural-Network Representation of
        High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401

    """

    para_pairs, para_triplets = parameters

    bp = BehlerParrinello()
    bp._unitcell = indices_unitcell

    # compute the distances and angles once for every structure
    distmatrix = cdist(coords, coords)
    cos_theta = bp.calculate_cosTheta(coords, distmatrix, periodic)

    # loops over different sets of parameters, overwriting the values
    # that were used to initialize the class
    g_1 = []
    for para in para_pairs:
        bp.r_cut, r_s, eta, lambda_, zeta = para
        g_1.append(bp.g_1(distmatrix, elements, periodic))
    g_2 = []
    for para in para_triplets:
        bp.r_cut, r_s, eta, lambda_, zeta = para
        g_2.append(bp.g_2(cos_theta, distmatrix, elements,
                          periodic))

    return (np.transpose(np.array(g_1), [2, 1, 0]),
            np.transpose(np.array(g_2), [2, 1, 0]))


def build_supercell(unitcell, R_c):
    """
    Build supercell for BP.

    Args:
        unitcell: Pymatgen IStructure object.
        R_c: Cutoff radius, in angstroms, for consideration.

    Returns:
        supercell: Pymatgen IStructure object.
        indices_unitcell: indices corresponding to original atom sites.

    """
    [n1, n2, n3] = [np.ceil(R_c / length) for length in unitcell.lattice.abc]
    structure = unitcell.copy()
    structure.translate_sites(indices=range(len(unitcell)),
                              vector=[-(n1 + 1), -(n2 + 1), -(n3 + 1)],
                              to_unit_cell=False)
    structure.make_supercell([2 * n1 + 1, 2 * n2 + 1, 2 * n3 + 1],
                             to_unit_cell=False)
    supercell = IStructure.from_sites(structure)

    index_search = np.isclose(supercell.cart_coords[:, None],
                              unitcell.cart_coords, atol=1e-4).all(-1)
    indices_unitcell = np.where(index_search.any(0), index_search.argmax(0),
                                None)
    indices_unitcell = [ind for ind in indices_unitcell if ind]

    return supercell, indices_unitcell


def dummy_fingerprint(s_data, parameters, system_symbols):
    """

    Args:
        s_data: List of data (output of read_structure).
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, symbol_set, species_counts,
     species_list, unit, periodic, property_value) = s_data
    assert set(symbol_set).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters
    n_atoms = len(coords)
    n_species = len(symbol_set)
    pair_num = len(list(combinations_with_replacement(range(n_species), 1)))
    triplet_num = len(list(combinations_with_replacement(range(n_species),
                                                         2)))

    coord_sums = np.sum(coords, axis=1)
    # sum of coordinates for each atom

    pair_factors = np.random.rand(pair_num, 1)
    # 1 factor per pair interaction (specie in sphere)
    para_factors_1 = np.sum(para_pairs, axis=0)
    para_num_1 = len(para_factors_1)
    # 1 factor per parameter set
    tile_atoms_1 = np.tile(coord_sums.reshape(n_atoms, 1, 1),
                           (1, pair_num, para_num_1))
    tile_inter_1 = np.tile(pair_factors.reshape(1, pair_num, 1),
                           (n_atoms, 1, para_num_1))
    tile_parameters_1 = np.tile(para_factors_1.reshape(1, 1, para_num_1),
                                (n_atoms, pair_num, 1))
    g_1 = np.multiply(np.multiply(tile_atoms_1, tile_inter_1),
                      tile_parameters_1)
    # g_1.shape ~ (#atoms x #species x #pair_parameters)
    triplet_factors = np.random.rand(triplet_num, 1)
    # 1 factor per triplet interaction (2 species in sphere)
    para_factors_2 = np.sum(para_triplets, axis=0)
    para_num_2 = len(para_factors_2)
    # 1 factor per parameter set
    tile_atoms_2 = np.tile(coord_sums.reshape(n_atoms, 1, 1),
                           (1, triplet_num, para_num_2))
    tile_inter_2 = np.tile(triplet_factors.reshape(1, triplet_num, 1),
                           (n_atoms, 1, para_num_2))
    tile_parameters_2 = np.tile(para_factors_2.reshape(1, 1, para_num_2),
                                (n_atoms, triplet_num, 1))
    g_2 = np.multiply(np.multiply(tile_atoms_2, tile_inter_2),
                      tile_parameters_2)
    # g_2.shape ~ (#atoms x
    #              [#combinations of species with replacement] x
    #              #triplet_parameters
    data = pad_fingerprints([g_1, g_2], symbol_set, system_symbols, [1, 2])
    labels = ['dummy_pairs', 'dummy_triplets']
    fingerprints = zip(labels, data)
    return fingerprints


def pad_fingerprints(terms, symbol_set, system_symbols, dims):
    """

    Args:
        terms: List of fingerprints.
        symbol_set: List of unique element names in structure as strings.
        system_symbols: List of system-wide unique element names as strings.
        dims: Dimensionality of interaction(s).
            (e.g. 1 for pairwise, 2 for triplets, [1,2] for both)

    Returns:
        padded: Padded fingerprints.

    """
    assert len(dims) == len(terms)
    symbol_order = {k: v for v, k in enumerate(system_symbols)}
    symbol_set = sorted(symbol_set, key=symbol_order.get)

    system_groups = [list(combinations_with_replacement(system_symbols, dim))
                     for dim in dims]

    s_groups = [list(combinations_with_replacement(symbol_set, dim))
                for dim in dims]

    n_groups = [len(groups) for groups in system_groups]

    padded = [np.zeros((g.shape[0], n_groups[j], g.shape[2]))
              for j, g in enumerate(terms)]

    content_slices = [[sys_group.index(group) for group in s_group]
                      for s_group, sys_group in zip(s_groups, system_groups)]

    for dim, content_indices in enumerate(content_slices):
        for s_index, sys_index in enumerate(content_indices):
            padded[dim][:, sys_index, :] = terms[dim][:, s_index, :]

    return padded
