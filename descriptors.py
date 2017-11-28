from builtins import range

import numpy, time, scipy.sparse

from molml.base import BaseFeature, SetMergeMixin
from molml.utils import get_element_pairs, get_index_mapping


class BehlerParrinello(SetMergeMixin, BaseFeature):
    
    """
    An implementation of the descriptors used in Behler-Parrinello Neural
    Networks.

    Parameters
    ----------
    input_type : string, default='list'
        Specifies the format the input values will be (must be one of 'list'
        or 'filename').

    n_jobs : int, default=1
        Specifies the number of processes to create when generating the
        features. Positive numbers specify a specifc amount, and numbers less
        than 1 will use the number of cores the computer has.

    r_cut : float, default=6.
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    r_s : float, default=1.0
        An offset parameter for computing gaussian values between pairwise
        distances.

    eta : float, default=1.0
        A decay parameter for the gaussian distances.

    lambda_ : float, default=1.0
        This value sets the orientation of the cosine function for the angles.
        It should only take values in {-1., 1.}.

    zeta : float, default=1.0
        A decay parameter for the angular terms.

    Attributes
    ----------
    _elements : set
        A set of all the elements in the molecules.

    _element_pairs : set
        A set of all the element pairs in the molecules.

    References
    ----------
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.
    
    """
   
    ATTRIBUTES = ("_elements", "_element_pairs")
    LABELS = ("_elements", "_element_pairs")

    def __init__(self, input_type='list', n_jobs=1, r_cut=6.0, r_s=1., eta=1.,
                 lambda_=1., zeta=1.):
        super(BehlerParrinello, self).__init__(input_type=input_type,
                                               n_jobs=n_jobs)
        self.r_cut = r_cut
        self.r_s = r_s
        self.eta = eta
        self.lambda_ = lambda_
        self.zeta = zeta
        self._elements = None
        self._element_pairs = None
        self._unitcell = None


    def fc(self, R):

        """
        A cutoff function:
    
            f_{R_{c}}(R_{ij}) = \begin{cases}
                0.5 ( \cos( \frac{\pi R_{ij}}{R_c} ) + 1 ), & R_{ij} \le R_c \\
                0,  & otherwise
            \end{cases}
    
        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
    
        Returns
        -------
        array, shape=(N_atoms, N_atoms)
        The new distance matrix with the cutoff function applied

        """
        
#        values = 1.0 + 0*R  ## for easy testing purposes only !! DO NOT USE in actual calculations !!
        values = 0.5 * (numpy.cos(numpy.pi * R / self.r_cut) + 1)
        values[R > self.r_cut] = 0
        values[R < 1E-08] = 0  ## in order to exclude "onsite" terms from sum 
        
        return values


    def cosTheta(self, coords, R, periodic=False):
        
        """
        Compute the angular term for all triples of atoms:

            cos(\Theta_{ijk}) = (R_{ij} . R_{ik}) / (|R_{ij}| |R_{ik}|)

        This is only a slight modification from molml.utils.get_angles

        Parameters
        ----------
        coords: array, shape=(N_atoms, 3)
                An array of the Cartesian coordinates of all the atoms  
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(N_centralatoms, N_atoms, N_atoms)
        The cosines triplet angles.
        cosTheta[i,j,k] is the angle of the triplet j-i-k (i is the central atom)
            
        """
        
        N_unit = self._N_unitcell
        
        R_vecs = coords - coords[:N_unit,None]
        with numpy.errstate(divide='ignore', invalid='ignore'):
            R_unitvecs = R_vecs / R[:N_unit,:,None]
        R_unitvecs = numpy.nan_to_num(R_unitvecs)
        ## the Einstein summation in the following line essentially performs the dot product Rij.Rik
        cosTheta = numpy.einsum('ijm,ikm->ijk', R_unitvecs, R_unitvecs)
        
        return cosTheta
    
    
    def G1(self, fc, R, elements, periodic=False):
        
        """
        A radial symmetry function:

            G^1_i = \sum_{j \neq i} \exp(- \eta (R_{ij} - R_s)^2) f_c(R_{ij})

        Parameters
        ----------
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(# atoms, # unique elements)
        The atom-wise G^1 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        elements = numpy.array(elements)
        
        values = numpy.exp(-self.eta * (R[:N_unit,:] - self.r_s) ** 2) * fc[:N_unit,:]       
        
        totals = []
        for ele in numpy.unique(sorted(elements)):
            ## find the positions of all atoms of type "ele"
            idxs = numpy.where(elements == ele)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            total = values[:, idxs].sum(axis=1)
            totals.append(total)
        
        return totals


    def G2(self, fc, cosTheta, R, elements, periodic=False):
        
        """
        An angular symmetry function:

            G^2_i = 2^{1-\zeta} \sum_{i,k \neq i}
                        (1 - \lambda \cos(\Theta_{ijk}))^\zeta
                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})

        Parameters
        ----------
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        cosTheta: array, shape=(N_atoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        R: array, shape=(N_atoms, N_atoms)
           A distance matrix for all the atoms (scipy.spatial.cdist).
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(# atoms, # unique element pairs)
        The atom-wise G^2 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        elements = numpy.array(elements)
        element_pairs = sorted(numpy.array(get_element_pairs(elements)),key=lambda x: (x[0],x[1]))
        
        values = (2 ** (1 - self.zeta) * (1 + self.lambda_ * cosTheta) ** self.zeta 
                  * numpy.exp(-self.eta * R[:N_unit,:,None]**2)
                  * numpy.exp(-self.eta * R[:N_unit,None,:]**2) 
                  * numpy.exp(-self.eta * R[None,:,:]**2)
                  * fc[:N_unit,:,None] * fc[:N_unit,None,:] * fc[None,:,:])

        totals = []
        for [ele1,ele2] in element_pairs:
            ## find the positions of all pairs of atoms of type (ele1,ele2)
            idxj,idxk = numpy.where(elements==ele1)[0],numpy.where(elements==ele2)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            total = (values[:, [[i] for i in idxj], idxk]).sum(axis=2).sum(axis=1)
            if ele1 != ele2:
                ## double the sum over pairs of (ele1,ele2) and (ele2,ele1)
                total = 2 * total
            totals.append(total)
            
        return totals
    

    def dRij_dRml(self, coords, R, periodic=False):
        
        """
        Computes the derivative of the norm of the position vector R_{ij}
        with respect to cartesian direction l of atom m.
    
        See Eq. 14c of the supplementary information of Khorshidi, Peterson, CPC(2016).
    
        Parameters
        ----------
        coords: list of [xyz] coords (in Angstroms)
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)
    
        Returns
        -------
        array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
        Access the derivatives by indexing dRij_dRml[i,j,m,l] which returns a scalar
        
        """
        
        N_unit = self._N_unitcell
        
        ## construct the identity matrices first with all atoms in the i and m dimensions
        ## then pick out the relevant slices in the i and m dimensions which correspond to "central" atoms
        deltajm_deltaim = (numpy.eye(len(coords))[None,:,:N_unit]
                            - numpy.eye(len(coords))[:,None,:N_unit])

        with numpy.errstate(divide='ignore', invalid='ignore'):
            R_inverse = 1./R
        R_inverse = numpy.nan_to_num(R_inverse)
        
        dRij_dRml = deltajm_deltaim[:,:,:,None] * (coords[None,:,None,:] - coords[:,None,None,:]) * R_inverse[:,:,None,None]
        
        return dRij_dRml


    def dRijvec_dRml(self, N_tot, periodic=False):
        
        """
        Computes the derivative of the position vector R_{ij}
        with respect to cartesian direction l of atom m.
    
        See Eq. 14d of the supplementary information of Khorshidi, Peterson, CPC(2016).
    
        Parameters
        ----------
        N_tot: total number of atoms
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)
    
        Returns
        -------
        array, shape=(N_centralatoms, N_atoms, N_centralatoms, 3, 3)
        Access the derivatives by indexing dRijvec_dRml[i,j,m,l] which returns a vector
        
        """

        N_unit = self._N_unitcell

        ## construct the identity matrices first with all atoms in the i and m dimensions
        ## then pick out the relevant slices in the i and m dimensions which correspond to "central" atoms
        deltajm_deltaim = (numpy.eye(N_tot)[:,:N_unit][None,:,:]
                            - numpy.eye(N_tot)[:N_unit,:N_unit][:,None,:])
        
        dRijvec_dRml = deltajm_deltaim[:,:,:,None,None] * numpy.eye(3)[None,None,None,:,:]
        
        return dRijvec_dRml


    def dfc_dRml(self, R, dRij_dRml):

        """
        The derivative of the cutoff function
        with respect to cartesian direction l of atom m:
    
            
        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
        dRij_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   An array of the derivatives of the distance matrix
    
        Returns
        -------
        array, shape=(N_atoms, N_atoms, N_centralatoms, 3)

        """
        
        values = 0.5 * (-numpy.pi / self.r_cut) * numpy.sin(numpy.pi * R / self.r_cut)[:,:,None,None] * dRij_dRml
        values[R > self.r_cut] = 0
        values[R < 1E-08] = 0  ## in order to exclude "onsite" terms from sum 
        
        return values
    
    
    def dcosTheta_dRml(self, coords, R, dRij_dRml, dRijvec_dRml, cosTheta, periodic=False):
        
        """
        See Eq. 14f of the supplementary information of Khorshidi, Peterson, CPC(2016).

        Parameters
        ----------
        coords: array, shape=(N_atoms, 3)
                An array of the Cartesian coordinates of all the atoms  
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        dRij_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   An array of the derivatives of the distance matrix
        dRijvec_dRml: array, shape=(N_centralatoms, N_atoms, N_centralatoms, 3, 3)
                      An array of the derivatives of the position vectors
        cosTheta: array, shape=(N_centralatoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(N_centralatoms, N_atoms, N_atoms, N_centralatoms, 3)
        The derivatives of the cosines of triplet angles.
        Access the derivatives by indexing dcosTheta_dRml[i,j,k,m,l] which returns a scalar
        
        """
        
        N_unit = self._N_unitcell
        
        R_vecs = coords - coords[:N_unit,None]

        dcosTheta_dRml = (numpy.einsum('ijmld,ikd->ijkml', dRijvec_dRml, R_vecs)
                                + numpy.einsum('ikmld,ijd->ijkml', dRijvec_dRml, R_vecs))

        ## the Einstein summation performs the dot products
        with numpy.errstate(divide='ignore', invalid='ignore'):
            dcosTheta_dRml = (dcosTheta_dRml
                                / (R[:N_unit,:,None,None,None]*R[:N_unit,None,:,None,None])
                                - (dRij_dRml[:N_unit,:,None,:,:] / R[:N_unit,:,None,None,None]
                                + dRij_dRml[:N_unit,None,:,:,:] / R[:N_unit,None,:,None,None])
                                * cosTheta[:,:,:,None,None])
        dcosTheta_dRml = numpy.nan_to_num(dcosTheta_dRml)

        return dcosTheta_dRml


    def dG1_dRml(self, fc, dfc_dRml, R, dRij_dRml, elements, periodic=False):
        
        """
        Derivative of G1 symmetry function.
        See Eq. 13b of the supplementary information of Khorshidi, Peterson, CPC(2016).

        Parameters
        ----------
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        dfc_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                  An array of the derivatives of fc.
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        dRij_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   An array of the derivatives of the distance matrix
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(N_atoms, N_elements)
        The atom-wise G^1 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        elements = numpy.array(elements)
        
        values = ((dRij_dRml[:N_unit,:,:,:] * (-2*self.eta * (R[:N_unit,:] - self.r_s) * fc[:N_unit,:])[:,:,None,None]
                    + dfc_dRml[:N_unit,:,:,:])
                    * numpy.exp(-self.eta * (R[:N_unit,:] - self.r_s) ** 2)[:,:,None,None])   
        
        totals = []
        for ele in numpy.unique(sorted(elements)):
            ## find the positions of all atoms of type "ele"
            idxs = numpy.where(elements == ele)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            total = values[:,idxs,:,:].sum(axis=1)
            totals.append(total)
        
        return totals


    def dG2_dRml(self, fc, dfc_dRml, cosTheta, dcosTheta_dRml, R, dRij_dRml, elements, periodic=False):
        
        """
        Derivative of G2 symmetry function.
        See Eq. 13d of the supplementary information of Khorshidi, Peterson, CPC(2016)

        Parameters
        ----------
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        dfc_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                  An array of the derivatives of fc.
        cosTheta: array, shape=(N_centralatoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        dcosTheta_dRml: array, shape=(N_centralatoms, N_atoms, N_atoms, N_centralatoms, 3)
                        The derivatives of the cosines of triplet angles.
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        dRij_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   An array of the derivatives of the distance matrix
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        array, shape=(# atoms, # unique element pairs)
        The atom-wise G^2 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        elements = numpy.array(elements)
        element_pairs = sorted(numpy.array(get_element_pairs(elements)),key=lambda x: (x[0],x[1])) 
        
        values = (2 ** (1 - self.zeta) * ((1 + self.lambda_ * cosTheta) ** (self.zeta - 1)
                  * numpy.exp(-self.eta * R[:N_unit,:,None]**2)
                  * numpy.exp(-self.eta * R[:N_unit,None,:]**2) 
                  * numpy.exp(-self.eta * R[None,:,:]**2))[:,:,:,None,None]
                  * (fc[:N_unit,:,None,None,None] * fc[:N_unit,None,:,None,None] * fc[None,:,:,None,None]
                     * (self.lambda_ * self.zeta * dcosTheta_dRml 
                        - 2*self.eta * (1 + self.lambda_ * cosTheta[:,:,:,None,None])
                         * (R[:N_unit,:,None,None,None] * dRij_dRml[:N_unit,:,None,:,:]
                          + R[:N_unit,None,:,None,None] * dRij_dRml[:N_unit,None,:,:,:]
                          + R[None,:,:,None,None] * dRij_dRml[None,:,:,:,:]))
                  + (1 + self.lambda_ * cosTheta[:,:,:,None,None])
                     * (dfc_dRml[:N_unit,:,None,:,:] * fc[:N_unit,None,:,None,None] * fc[None,:,:,None,None] 
                      + dfc_dRml[:N_unit,None,:,:,:] * fc[:N_unit,:,None,None,None] * fc[None,:,:,None,None]
                      + dfc_dRml[None,:,:,:,:] * fc[:N_unit,:,None,None,None] * fc[:N_unit,None,:,None,None])))
                           
        totals = []
        for [ele1,ele2] in element_pairs:
            ## find the positions of all pairs of atoms of type (ele1,ele2)
            idxj,idxk = numpy.where(elements==ele1)[0],numpy.where(elements==ele2)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            total = (values[:, [[i] for i in idxj], idxk]).sum(axis=2).sum(axis=1)
            if ele1 != ele2:
                ## double the sum over pairs of (ele1,ele2) and (ele2,ele1)
                total = 2 * total
            totals.append(total)
            
        return totals
    
    