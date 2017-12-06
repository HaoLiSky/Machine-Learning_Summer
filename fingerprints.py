"""
Fingerprint functions.
"""
import numpy as np
import ase
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations_with_replacement
from ase.build.supercells import make_supercell


def bp_fingerprint(s_data, parameters, system_elements, derivs=False):
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
        system_elements: List of system-wide unique element names as strings.
                         Given as an input to the command line.
        derivs (boolean): Whether to calculate derivatives of fingerprints
            with respect to cartesian coordinates.

    """
    
    (coords, elements_set, element_counts,
     element_list, unit_cell, periodic, property_value) = s_data
    if not set(elements_set).issubset(set(system_elements)):
        raise AssertionError(str('-'.join(set(elements_set))),
                             'not valid for',
                             str('-'.join(set(system_elements))))

    if not len(element_list) == len(coords):
        print(element_list, coords.shape)
    assert len(element_list) == len(coords)
    assert set(elements_set).issubset(set(system_elements))

    if periodic:
        unitcell = ase.Atoms(''.join(element_list),
                             positions=coords,
                             cell=unit_cell)

        # generate supercell to include all neighbors of atoms in unitcell
        coords, element_list, N_unitcell = build_supercell(unitcell, R_c=6.0)

        g_list, g_orders = represent_BP(np.asarray(coords),np.asarray(element_list),
                                        parameters,derivs=False, 
                                        periodic=True,N_unitcell=N_unitcell)
    else:
        g_list, g_orders = represent_BP(np.asarray(coords),np.asarray(element_list),
                                        parameters,derivs=False)

    data = pad_fingerprints_by_interaction(g_list, elements_set,
                                           system_elements, g_orders)
    labels = ['G_1', 'G_2']
    if derivs:
        labels += ['dG_1', 'dG_2']
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
                                [0, 0, 2 * n3 + 1]])

    ## wrap supercell so that original unitcell is in the center
    supercell.wrap(center=(0.5 / (2 * n1 + 1),
                           0.5 / (2 * n2 + 1),
                           0.5 / (2 * n3 + 1)))

    coords1, elements1 = supercell.get_positions() ,supercell.get_chemical_symbols()

    ## sort atoms so that atoms in the original unitcell are listed first in the supercell
    ## and trim away atoms that are not within cutiff distance of any atoms in the unitcell
    coords, elements = unitcell.get_positions().tolist(), unitcell.get_chemical_symbols()
    R = cdist(unitcell.get_positions(), np.asarray(coords1))
    for j in range(len(coords1)):
        if np.any(np.less(R[:,j],1E-08)):
            ## if atom in unitcell, do not add to the list
            continue
        elif np.any(np.less_equal(R[:,j],R_c)):
            ## if atom is within cutoff distance of an atom in the unitcell, add to the list
            coords.append(coords1[j])
            elements.append(elements1[j])        
                
    return coords, elements, len(unitcell.positions)


def represent_BP(coords, elements, system_symbols, parameters=None, derivs=False,
                 periodic=False, N_unitcell=None):
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
    derivs: calculate derivatives of fingerprints.
    periodic: boolean (False = cluster/molecule, True = 3D periodic structure)
    N_unitcell: number of atoms in the unitcell
                (only applicable for periodic structures)

    Returns
    -------
    fingerprints:
        (# atoms, # unique element types, # descriptors) array of G^1s and 
        (# atoms, # unique element type pairs, # descriptors) array of G^2s.
        If derivs=True, also returns
        (# atoms, # atoms, # cart directions(3), # unique element types, # descriptors) array of dG^1/dRs and 
        (# atoms, # atoms, # cart directions(3), # unique element type pairs, # descriptors) array of dG^2/dRs.
    interaction_dims: Dimensionality of interaction(s).
        (e.g. 1 for pairwise, 2 for triplets, [1,2] for both)

    Notes
    -----
    Behler-Parrinello symmetry functions as described in:
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401

    """

    para_pairs, para_triplets = parameters

    bp = BehlerParrinello()
    
    bp._elements = elements
    bp._element_order = {k: v for v, k in enumerate(system_symbols)}
    bp._elements_unique = sorted(np.unique(elements), key=bp._element_order.get)
    bp._element_pairs = list(combinations_with_replacement(bp._elements_unique,2))

    if N_unitcell == None:
        bp._N_unitcell = len(coords)
    else:
        bp._N_unitcell = N_unitcell
#    print (N_unitcell,len(coords))

    ## quantities which are only computed once for every structure
    ## if r_c is fixed, then we can compute the cutoff functions here as well...
    R = cdist(coords, coords)
    fc = bp.fc(R)
    cosTheta = bp.cosTheta(coords, R, periodic = periodic)
    G1s,G2s = [],[]
    
    if derivs:
        dRij_dRml, Rij_dRij_dRml = bp.dRij_dRml(coords, R, periodic = periodic)
        Rij_dRij_dRml_sum = bp.Rij_dRij_dRml_sum(Rij_dRij_dRml)
        dfc_dRml = bp.dfc_dRml(R, dRij_dRml)
        fcinv_dfc_dRml_sum = bp.fcinv_dfc_dRml_sum(fc, dfc_dRml)
        dcosTheta_dRml = bp.dcosTheta_dRml(coords, R, dRij_dRml, cosTheta, periodic = periodic)
        dG1s,dG2s = [],[]
        
        
    ## loops over different sets of parameters
    for para in para_pairs:
        bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
#        fc = bp.fc(R)
#        dfc_dRml = bp.dfc_dRml(R, dRij_dRml)
        G1s.append(bp.G1(R, fc, periodic = periodic))
        if derivs:
            dG1s.append(bp.dG1_dRml(R, dRij_dRml, fc, dfc_dRml, periodic = periodic))
 
    for para in para_triplets:
        bp.r_cut, bp.r_s, bp.eta, bp.zeta, bp.lambda_ = para
#        fc = bp.fc(R)
#        dfc_dRml = bp.dfc_dRml(R, dRij_dRml)
#        fcinv_dfc_dRml_sum = bp.fcinv_dfc_dRml_sum(fc, dfc_dRml)
        G2,G2vals = bp.G2(R, fc, cosTheta, periodic = periodic)
        G2s.append(G2)
        if derivs:
            dG2s.append(bp.dG2_dRml(Rij_dRij_dRml_sum, fcinv_dfc_dRml_sum, cosTheta, dcosTheta_dRml, G2vals, periodic = periodic))

    fingerprints = [np.transpose(np.array(G1s), [2, 1, 0]),
                    np.transpose(np.array(G2s), [2, 1, 0])]
    interaction_dims = [1, 2]

    if derivs:
        fingerprints += [np.transpose(np.array(dG1s), [2, 3, 4, 1, 0]),
                         np.transpose(np.array(dG2s), [2, 3, 4, 1, 0])]
        interaction_dims += [1, 2]

    return fingerprints, interaction_dims


def dummy_fingerprint(s_data, parameters, system_symbols):
    """

    Args:
        s_data: List of data (output of read_collated_structure).
        parameters: Descriptor parameters.
        system_symbols: List of system-wide unique element names as strings.

    """
    (coords, structure_symbols, species_counts,
     species_list, unit, periodic, property_value) = s_data
    assert set(structure_symbols).issubset(set(system_symbols))
    para_pairs, para_triplets = parameters

    n_atoms = len(coords)
    pair_num = len(list(combinations_with_replacement(structure_symbols, 1)))
    triplet_num = len(list(combinations_with_replacement(structure_symbols,2)))
    para_num_1 = para_pairs.shape[0]
    para_num_2 = para_triplets.shape[0]

    g_1 = np.zeros(shape=(n_atoms, pair_num, para_num_1))
    g_2 = np.zeros(shape=(n_atoms, triplet_num, para_num_2))

    # g_1.shape ~ (#atoms x #species x #pair_parameters)
    # g_2.shape ~ (#atoms x
    #              [#combinations of species with replacement] x
    #              #triplet_parameters
    data = pad_fingerprints_by_interaction([g_1, g_2], structure_symbols,
                                           system_symbols, [1, 2])
    labels = ['dummy_pairs', 'dummy_triplets']
    fingerprints = zip(labels, data)
    return fingerprints, [x.shape for x in data]


def pad_fingerprints_by_interaction(terms, symbol_set, system_symbols, interaction_dims):
    """

    Args:
        terms: List of fingerprints.
        symbol_set: List of unique element names in structure as strings.
        system_symbols: List of system-wide unique element names as strings.
        interaction_dims: Dimensionality of interaction(s).
            (e.g. 1 for pairwise, 2 for triplets, [1,2] for both)

    Returns:
        padded: Padded fingerprints.

    """
    assert len(interaction_dims) == len(terms)
    symbol_order = {k: v for v, k in enumerate(system_symbols)}
    symbol_set = sorted(symbol_set, key=symbol_order.get)

    system_groups = [list(combinations_with_replacement(system_symbols, dim))
                     for dim in interaction_dims]

    s_groups = [list(combinations_with_replacement(symbol_set, dim))
                for dim in interaction_dims]

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
