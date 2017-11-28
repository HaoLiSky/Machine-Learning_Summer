"""
Fingerprint functions.
"""
import numpy as np
import ase
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
# from pymatgen.core.structure import Structure, IStructure
from ase.build.supercells import make_supercell


def bp_fingerprint(s_data, parameters, sys_elements, primes=False):
    """

    Parrinello-Behler representation. Computes fingerprints for
    pairwise (g_1) and triple (g_2) interactions.

    Fingerprints are ndarrays of size (n x s x k)
    where n = number of atoms, k = number of parameters given for the
    fingerprint, and s = combinations with replacement of the system's species
    set. When the input's species set is less than the system's species set,
    fingerprints are padded.

    Args:
        s_data: List of data (output of read_collated_structure).
        parameters: Descriptor parameters.
        sys_elements: List of system-wide unique element names as strings.
        primes (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.

    """
    (coords, elements_set, element_counts,
     element_list, unit_cell, periodic, property_value) = s_data
    if not set(elements_set).issubset(set(sys_elements)):
        raise AssertionError(str('-'.join(set(elements_set))),
                             'not valid for',
                             str('-'.join(set(sys_elements))))

    if not len(element_list) == len(coords):
        print(element_list, coords.shape)
    assert len(element_list) == len(coords)
    assert set(elements_set).issubset(set(sys_elements))

    if periodic:
        unitcell = ase.Atoms(''.join(element_list),
                             positions=coords,
                             cell=unit_cell)

        # generate supercell to include all neighbors of atoms in unitcell
        supercell, N_unitcell = build_supercell(unitcell, R_c=6.0)

        g_list, g_orders = represent_BP(np.asarray(supercell.positions),
                                        supercell.get_chemical_symbols(),
                                        parameters, periodic=True,
                                        N_unitcell=N_unitcell,
                                        primes=primes)
    else:
        g_list, g_orders = represent_BP(coords, element_list,
                                        parameters)

    data = pad_fingerprints_by_interaction(g_list, elements_set,
                                           sys_elements, g_orders)
    labels = ['G_1', 'G_2']
    if primes:
        labels += ['dG_1', 'dG2_']
    fingerprints = zip(labels, data)
    return fingerprints, [x.shape for x in data]


def build_supercell(unitcell, R_c):
    """
    build supercell from a given unitcell

    Parameters
    ----------
    unitcell: ASE atoms object for the unitcell
    R_c: cutoff distance (in Angstroms)

    Returns
    -------
    supercell: ASE atoms object for the supercell
    number of atoms in the unitcell

    """

    [n1, n2, n3] = [np.ceil(R_c / length) for length in
                    unitcell.get_cell_lengths_and_angles()[:3]]
    supercell = make_supercell(unitcell,
                               [[2 * n1 + 1, 0, 0],
                                [0, 2 * n2 + 1, 0],
                                [0, 0, 2 * n2 + 1]])
    supercell.wrap(
        center=(0.5 / (2 * n1 + 1), 0.5 / (2 * n2 + 1), 0.5 / (2 * n3 + 1)))

    assert np.all(np.isclose(supercell.get_positions()[:len(unitcell), ...],
                             unitcell.get_positions()))

    return supercell, len(unitcell.positions)


def represent_BP(coords, elements, parameters=None, periodic=False,
                 N_unitcell=None, primes=False):
    """
    computes the Behler-Parrinello atom-based descriptors for each atom in a given structure

    Parameters
    ----------
    coords: list of [xyz] coords (in Angstroms)
    elements: list of element name strings
    parameters: list of parameters for Behler-Parrinello symmetry functions
        r_cut: cutoff radius (angstroms); default = 6.0
        r_s: pairwise distance offset (angstroms); default = 1.0
        eta: exponent term dampener; default = 1.0
        lambda_: angular expansion switch +1 or -1; default = 1.0
        zeta: angular expansion degree; default = 1.0
    periodic: boolean (False = cluster/molecule, True = 3D periodic structure)
    N_unitcell: number of atoms in the unitcell
                (only applicable for periodic structures)
    primes: calculate derivatives of fingerprints.

    Returns
    -------
    A (# atoms, # unique element types, # descriptors) array of G^1s and
    a (# atoms, # unique element type pairs, # descriptors) array of G^2s.

    Notes
    -----
    Behler-Parrinello symmetry functions as described in:
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401

    """

    para_pairs, para_triplets = parameters

    bp = BehlerParrinello()

    if N_unitcell:
        bp._N_unitcell = N_unitcell
    else:
        bp._N_unitcell = len(coords)

    ## we only compute the distances and angles once for every structure
    distmatrix = cdist(coords, coords)
    fc = bp.fc(distmatrix)
    cosTheta = bp.cosTheta(coords, distmatrix, periodic=periodic)

    ## loops over different sets of parameters, overwriting the values that were used to initialize the class
    G1 = []
    for para in para_pairs:
        bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
        G1.append(bp.G1(fc, distmatrix, elements, periodic=periodic))

    G2 = []
    for para in para_triplets:
        bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
        G2.append(bp.G2(fc, cosTheta, distmatrix, elements, periodic=periodic))

    fingerprints = [np.transpose(np.array(G1), [2, 1, 0]),
                    np.transpose(np.array(G2), [2, 1, 0])]
    orders = [1, 2]

    if primes:
        dRij_dRml = bp.dRij_dRml(coords, distmatrix, periodic=periodic)
        dRijvec_dRml = bp.dRijvec_dRml(len(coords), periodic=periodic)
        dfc_dRml = bp.dfc_dRml(distmatrix, dRij_dRml)
        dcosTheta_dRml = bp.dcosTheta_dRml(coords, distmatrix, dRij_dRml,
                                           dRijvec_dRml, cosTheta,
                                           periodic=periodic)
        dG1 = []
        for para in para_pairs:
            bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
            dG1.append(
                bp.dG1_dRml(fc, dfc_dRml, distmatrix, dRij_dRml, elements,
                            periodic=periodic))

        dG2 = []
        for para in para_triplets:
            bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
            dG2.append(
                bp.dG2_dRml(fc, dfc_dRml, cosTheta, dcosTheta_dRml, distmatrix,
                            dRij_dRml,
                            elements,
                            periodic=periodic))

        fingerprints += [np.transpose(np.array(dG1), [2, 3, 4, 1, 0]),
                         np.transpose(np.array(dG2), [2, 3, 4, 1, 0])]
        orders += [1, 2]

    return fingerprints, orders


def dummy_fingerprint(s_data, parameters, system_symbols):
    """

    Args:
        s_data: List of data (output of read_collated_structure).
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, symbol_set, species_counts,
     species_list, unit, periodic, property_value) = s_data
    assert set(symbol_set).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters

    n_atoms = len(coords)
    pair_num = len(list(combinations_with_replacement(symbol_set, 1)))
    triplet_num = len(list(combinations_with_replacement(symbol_set,
                                                         2)))
    para_num_1 = para_pairs.shape[0]
    para_num_2 = para_triplets.shape[0]

    g_1 = np.zeros(shape=(n_atoms, pair_num, para_num_1))
    g_2 = np.zeros(shape=(n_atoms, triplet_num, para_num_2))

    # g_1.shape ~ (#atoms x #species x #pair_parameters)
    # g_2.shape ~ (#atoms x
    #              [#combinations of species with replacement] x
    #              #triplet_parameters
    data = pad_fingerprints_by_interaction([g_1, g_2], symbol_set,
                                           system_symbols, [1, 2])
    labels = ['dummy_pairs', 'dummy_triplets']
    fingerprints = zip(labels, data)
    return fingerprints, [x.shape for x in data]


def pad_fingerprints_by_interaction(terms, symbol_set, system_symbols, dims):
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

    group_deltas = [len(groups_f) - len(groups_i)
                    for groups_i, groups_f in zip(s_groups, system_groups)]
    #  difference in number of alpha indices in structure vs system

    pad_widths = [[(0, 0)] * (len(term.shape) - 2)
                  + [(padding, 0)]  # only pad alpha dimension
                  + [(0, 0)]
                  for term, padding in zip(terms, group_deltas)]

    padded = [np.pad(np.zeros(data.shape), pad_width, 'edge')
              for data, pad_width in zip(terms, pad_widths)]

    content_slices = [[sys_group.index(group) for group in s_group]
                      for s_group, sys_group in zip(s_groups, system_groups)]

    for dim, content_indices in enumerate(content_slices):
        for s_index, sys_index in enumerate(content_indices):
            padded[dim][..., sys_index, :] = terms[dim][..., s_index, :]

    return padded
