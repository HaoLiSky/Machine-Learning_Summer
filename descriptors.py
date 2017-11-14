from builtins import range

import numpy, time

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


    def f_c(self, R):

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
        values : array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied

        """
        
#        values = 1.0 + 0*R  ## for easy testing purposes only !! DO NOT USE in actual calculations !!
        values = 0.5 * (numpy.cos(numpy.pi * R / self.r_cut) + 1)
        values[R > self.r_cut] = 0
        values[R < 1E-08] = 0  ## in order to exclude "onsite" terms from sum 
        
        return values
    
    
    def g_1(self, R, elements, periodic):
        
        """
        A radial symmetry function:

            G^1_i = \sum_{j \neq i} \exp(- \eta (R_{ij} - R_s)^2) f_c(R_{ij})

        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        total : array, shape=(N_atoms, N_elements)
            The atom-wise g_1 evaluations.
            
        """
        
        if periodic == True: 
            R = R[self._unitcell,:]
        
        elements = numpy.array(elements)
        
        values = numpy.exp(-self.eta * (R - self.r_s) ** 2) * self.f_c(R)
                
        totals = []
        for ele in numpy.unique(sorted(elements)):
            ## find the positions of all atoms of type "ele"
            idxs = numpy.where(elements == ele)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            total = values[:, idxs].sum(axis=1)
            totals.append(total)
        
        return numpy.array(totals).T


    def g_2(self, cosTheta, R, elements, periodic):
        
        """
        An angular symmetry function:

            G^2_i = 2^{1-\zeta} \sum_{i,k \neq i}
                        (1 - \lambda \cos(\Theta_{ijk}))^\zeta
                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})

        This function needs to be optimized.

        Parameters
        ----------
        cosTheta: array, shape=(N_atoms, N_atoms, N_atoms)
                    An array of cosines of triplet angles.
        R: array, shape=(N_atoms, N_atoms)
           A distance matrix for all the atoms (scipy.spatial.cdist).
        elements: list of element name strings
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        total : array, shape=(N_atoms, len(self._element_pairs))
                The atom-wise g_2 evaluations.
            
        """
        
        elements = numpy.array(elements)
        element_pairs = sorted(numpy.array(get_element_pairs(elements)),key=lambda x: (x[0],x[1]))
        
        if periodic == True:
            i_indices = self._unitcell
        elif periodic == False:
            i_indices = [i for i in range(len(elements))]
        
        values = (self.f_c(R[i_indices,:,None]) * self.f_c(R[i_indices,None,:]) * self.f_c(R[None,:,:])
                * numpy.exp(-self.eta * (R[i_indices,:,None]**2 + R[i_indices,None,:]**2 + R[None,:,:]**2)))
                
        values = 2 ** (1 - self.zeta) * (1 - self.lambda_ * cosTheta) ** self.zeta * values

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
    

#    def g_2(self, cosTheta, R, elements):
#        
#        """
#        An angular symmetry function:
#
#            G^2_i = 2^{1-\zeta} \sum_{i,k \neq i}
#                        (1 - \lambda \cos(\Theta_{ijk}))^\zeta
#                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
#                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})
#
#        This function needs to be optimized.
#
#        Parameters
#        ----------
#        cosTheta : array, shape=(N_atoms, N_atoms, N_atoms)
#            An array of cosines of triplet angles.
#
#        R : array, shape=(N_atoms, N_atoms)
#            A distance matrix for all the atoms (scipy.spatial.cdist).
#
#        elements : list
#            A list of all the elements in the molecule.
#
#        Returns
#        -------
#        total : array, shape=(N_atoms, len(self._element_pairs))
#            The atom-wise g_2 evaluations.
#            
#        """
#        
#        F_c_R = self.f_c(R)
#
#        R2 = self.eta * R ** 2
#        new_Theta = (1 - self.lambda_ * cosTheta) ** self.zeta
#
#        get_index, length, _ = get_index_mapping(get_element_pairs(elements),2,False)
#
#        n = R.shape[0]
#        values = numpy.zeros((n, length))
#        for i in range(n):
#            for j in range(n):
#                if not F_c_R[i,j] or i == j:
#                    continue
#                ele1 = elements[j]
#
#                for k in range(n):
#                    if k <= j or not F_c_R[i,k] or not F_c_R[j,k] or k == i:
#                        continue
#                    ele2 = elements[k]
#                    eles = ele1, ele2
#
#                    temp = new_Theta[i,j,k] * numpy.exp(-(R2[i,j] + R2[i,k] + R2[j,k])) * F_c_R[i,j] * F_c_R[i,k] * F_c_R[j,k]
#                    try:
#                        values[i, get_index(eles)] += 2*temp
#                    except KeyError:
#                        pass
#        
#        return 2 ** (1 - self.zeta) * values


#    def calculate_cosTheta(self, coords, R):
#        
#        """
#        Compute the angular term for all triples of atoms:
#
#            cos(\Theta_{ijk}) = (R_{ij} . R_{ik}) / (|R_{ij}| |R_{ik}|)
#
#        Right now this is a fairly naive implementation so this could be
#        optimized quite a bit.
#
#        Parameters
#        ----------
#        coords : array, shape=(N_atoms, 3)
#            An array of the Cartesian coordinates of all the atoms
#        
#        R : array, shape=(N_atoms, N_atoms)
#            A distance matrix for all the atoms (scipy.spatial.cdist).
#
#        Returns
#        -------
#        Theta : array, shape=(N_atoms, N_atoms, N_atoms)
#            The angular term for all the atoms given.
#            
#        """
#        
#        n = coords.shape[0]
#        cosTheta = numpy.zeros((n, n, n))
#        for i, Ri in enumerate(coords):
#            for j, Rj in enumerate(coords):
#                if i == j:
#                    continue
#                Rij = Ri - Rj
#                for k, Rk in enumerate(coords):
#                    if j <= k or i == k:
#                        continue
#                    Rik = Ri - Rk
#                    cosTheta_ijk = numpy.dot(Rij, Rik) / (R[i,j] * R[i,k])
#                    cosTheta[i,j,k] = cosTheta_ijk 
#                    cosTheta[i,k,j] = cosTheta_ijk 
#                    
#        return cosTheta
    
    
    def calculate_cosTheta(self, coords, R, periodic):
        
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
        cosTheta: array, shape=(N_atoms, N_atoms, N_atoms)
                  The angular terms for all triplets of atoms.
                  cosTheta[i,j,k] is the angle of the triplet j-i-k (i is the central atom)
            
        """
        
        if periodic == False:
            coordsi = coords
        elif periodic == True: 
            coordsi = coords[self._unitcell]
            R = R[self._unitcell,:]
        
        R_vecs = coords - coordsi[:,None]
        with numpy.errstate(divide='ignore', invalid='ignore'):
            R_unitvecs = R_vecs / R[:,:,None]
        R_unitvecs = numpy.nan_to_num(R_unitvecs)
        ## the Einstein summation in the following line essentially performs the dot product Rij.Rik
        cosTheta = numpy.einsum('ijm,ikm->ijk', R_unitvecs, R_unitvecs)
            
        return cosTheta

