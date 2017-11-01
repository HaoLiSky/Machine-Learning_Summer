from builtins import range

import numpy, time
from scipy.spatial.distance import cdist

from molml.base import BaseFeature, SetMergeMixin
from molml.utils import get_element_pairs, cosine_decay, get_angles
from molml.utils import get_index_mapping


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


    def f_c(self, R):
        
        return cosine_decay(R, r_cut=self.r_cut)
    
    
    def g_1(self, R, elements):
        
        """
        A radial symmetry function:

            G^1_i = \sum_{j \neq i} \exp(- \eta (R_{ij} - R_s)^2) f_c(R_{ij})

        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)

        Returns
        -------
        total : array, shape=(N_atoms, N_elements)
            The atom-wise g_1 evaluations.
            
        """
        
        values = numpy.exp(-self.eta * (R - self.r_s) ** 2) * self.f_c(R)
        numpy.fill_diagonal(values, 0)

        elements = numpy.array(elements)

        totals = []
        for ele in sorted(self._elements):
            idxs = numpy.where(elements == ele)[0]
            total = values[:, idxs].sum(1)
            totals.append(total)
            
        return numpy.array(totals).T


    def g_2(self, cosTheta, R, elements):
        
        """
        An angular symmetry function:

            G^2_i = 2^{1-\zeta} \sum_{i,k \neq i}
                        (1 - \lambda \cos(\Theta_{ijk}))^\zeta
                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})

        This function needs to be optimized.

        Parameters
        ----------
        cosTheta : array, shape=(N_atoms, N_atoms, N_atoms)
            An array of cosines of triplet angles.

        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).

        elements : list
            A list of all the elements in the molecule.

        Returns
        -------
        total : array, shape=(N_atoms, len(self._element_pairs))
            The atom-wise g_2 evaluations.
            
        """
        
        F_c_R = self.f_c(R)

        R2 = self.eta * R ** 2
#        new_Theta = (1 - self.lambda_ * numpy.cos(cosTheta)) ** self.zeta
        new_Theta = (1 - self.lambda_ * cosTheta) ** self.zeta

        get_index, length, _ = get_index_mapping(self._element_pairs,2,False)

        n = R.shape[0]
        values = numpy.zeros((n, length))
        for i in range(n):
            for j in range(n):
                if not F_c_R[i, j] or i == j:
                #if i == j or not F_c_R[i, j]:
                    continue
                ele1 = elements[j]

                for k in range(n):
                    if k <= j or not F_c_R[i, k] or not F_c_R[j, k] or k == i:
                        continue
                    #if k == i or j == k:
                    #    continue
                    #if not F_c_R[i, k] or not F_c_R[j, k]:
                    #    continue
                    ele2 = elements[k]
                    eles = ele1, ele2

                    exp_term = numpy.exp(-(R2[i, j] + R2[i, k] + R2[j, k]))
                    angular_term = new_Theta[i, j, k]
                    radial_cuts = F_c_R[i, j] * F_c_R[i, k] * F_c_R[j, k]
                    temp = angular_term * exp_term * radial_cuts
                    try:
                        values[i, get_index(eles)] += 2*temp
#                        values[i, get_index(eles)] += temp
                    except KeyError:
                        pass
                    
        return 2 ** (1 - self.zeta) * values


    def calculate_cosTheta(self, R_vecs):
        
        """
        Compute the angular term for all triples of atoms:

            cos(\Theta_{ijk}) = (R_{ij} . R_{ik}) / (|R_{ij}| |R_{ik}|)

        Right now this is a fairly naive implementation so this could be
        optimized quite a bit.

        Parameters
        ----------
        R_vecs : array, shape=(N_atoms, 3)
            An array of the Cartesian coordinates of all the atoms

        Returns
        -------
        Theta : array, shape=(N_atoms, N_atoms, N_atoms)
            The angular term for all the atoms given.
            
        """
        
#        t0 = time.time()
        n = R_vecs.shape[0]
        Theta = numpy.zeros((n, n, n))
        for i, Ri in enumerate(R_vecs):
            for j, Rj in enumerate(R_vecs):
                if i == j:
                    continue
                Rij = Ri - Rj
                normRij = numpy.linalg.norm(Rij)
                for k, Rk in enumerate(R_vecs):
                    if j <= k or i == k:
                        continue
                    Rik = Ri - Rk
                    normRik = numpy.linalg.norm(Rik)
                    cosTheta = numpy.dot(Rij, Rik) / (normRij * normRik)
                    Theta[i, j, k] = cosTheta
                    Theta[i, k, j] = cosTheta
#        print ("calc theta time: ",time.time()-t0)            
        return Theta

