import numpy, time
from scipy import sparse
#from numba import jit

class BehlerParrinello():
    
    """
    An implementation of the Behler-Parrinello descriptors. 

    Parameters
    ----------
    r_cut : float, default=6.
        The maximum distance allowed for atoms to be considered local to the
        "central atom".

    r_s : float, default=0.0
        An offset parameter for computing gaussian values between pairwise
        distances.

    eta : float, default=1.0
        A decay parameter for the gaussian distances.

    zeta : float, default=1.0
        A decay parameter for the angular terms.
        
    lambda_ : float, default=1.0
        This value sets the orientation of the cosine function for the angles.
        It should only take values of +1 or -1.

    Attributes
    ----------
    _elements : list
        A list of all the elements in the molecules.

    _element_pairs : list
        A list of all unique element pairs in the molecules.
        
    _N_unitcell : int
        Number of atoms in the unitcell.

    References
    ----------
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.
    
    """
   
    ATTRIBUTES = ("_elements", "_element_pairs", "_N_unitcell")
    LABELS = ("_elements", "_element_pairs", "_N_unitcell")

    def __init__(self, r_cut=6.0, r_s=0., eta=1., lambda_=1., zeta=1.):
        self.r_cut = r_cut
        self.r_s = r_s
        self.eta = eta
        self.zeta = zeta
        self.lambda_ = lambda_
        self._system_elements = None
        self._elements = None
        self._element_pairs = None
        self._N_unitcell = None


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
        fc : array, shape=(N_atoms, N_atoms)
             The new distance matrix with the cutoff function applied

        """
        
        fc = 0.5 * (numpy.cos(numpy.pi * R / self.r_cut) + 1)
        fc [R > self.r_cut] = 0
        fc[R < 1E-08] = 0  ## in order to exclude self terms from sum 
        
        return fc


    def cosTheta(self, coords, R, periodic=False):
        
        """
        Compute the angular term for all triples of atoms:

            cos(\theta_{ijk}) = (R_{ij} . R_{ik}) / (|R_{ij}| |R_{ik}|)

        Parameters
        ----------
        coords: array, shape=(N_atoms, 3)
                An array of the Cartesian coordinates of all the atoms  
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        cosTheta : array, shape=(N_centralatoms, N_atoms, N_atoms)
                   The cosines triplet angles.
                   cosTheta[i,j,k] is the angle of the triplet j-i-k (i is the central atom)
            
        """
        
        N_unit = self._N_unitcell
        
        R_vecs = coords - coords[:N_unit,None]
        with numpy.errstate(divide='ignore', invalid='ignore'):
            R_unitvecs = numpy.divide(R_vecs,R[:N_unit,:,None])
        R_unitvecs = numpy.nan_to_num(R_unitvecs)
        
        ## the Einstein summation in the following line essentially performs the dot product Rij.Rik
        cosTheta = numpy.einsum('ijm,ikm->ijk', R_unitvecs, R_unitvecs)                  
        
        return cosTheta
    
    
    def G1(self, R, fc, periodic=False):
        
        """
        Radial symmetry function:

            G^1_i = \sum_{j \neq i} \exp(- \eta (R_{ij} - R_s)^2) f_c(R_{ij})

        Parameters
        ----------d
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist)
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applie
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        G1 : array, shape=(N_centralatoms, N_elements)
             The atom-wise G^1 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        values = numpy.exp(-self.eta * (R[:N_unit,:] - self.r_s) ** 2) * fc[:N_unit,:]
        
        G1 = []
        for ele in self._elements_unique:
            ## find the positions of all atoms of type "ele"
            idxs = numpy.where(self._elements == ele)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            G1.append(values[:, idxs].sum(axis=1))
        
        return G1


    def G2(self, R, fc, cosTheta, periodic=False):
        
        """
        Angular symmetry function:

            G^2_i = 2^{1-\zeta} \sum_{j,k \neq i}
                        (1 + \lambda \cos(\theta_{ijk}))^\zeta
                        \exp(-\eta (R_{ij}^2 + R_{ik}^2 + R_{jk}^2))
                        f_c(R_{ij}) f_c(R_{ik}) f_c(R_{jk})

        Parameters
        ----------
        R: array, shape=(N_atoms, N_atoms)
           A distance matrix for all the atoms (scipy.spatial.cdist).
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        cosTheta: array, shape=(N_atoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        G2 : array, shape=(N_centralatoms, N_elementpairs)
             The atom-wise G^2 evaluations.
        values : array, shape=(N_centralatoms, N_atoms, N_atoms)
                 The angular terms inside the sum.
                 (To be reused when evaluating dG2/dRml)
            
        """
        
        N_unit = self._N_unitcell
       
        values = (2 ** (1 - self.zeta) * (1 + self.lambda_ * cosTheta) ** self.zeta 
                  * numpy.exp(-self.eta * R[:N_unit,:,None]**2)
                  * numpy.exp(-self.eta * R[:N_unit,None,:]**2) 
                  * numpy.exp(-self.eta * R[None,:,:]**2)
                  * fc[:N_unit,:,None] * fc[:N_unit,None,:] * fc[None,:,:])

        G2 = []
        for [ele1,ele2] in self._element_pairs:
            ## find the positions of all pairs of atoms of type (ele1,ele2)
            idxj,idxk = numpy.where(self._elements==ele1)[0],numpy.where(self._elements==ele2)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            if ele1 != ele2:
                ## double the sum over pairs of (ele1,ele2) and (ele2,ele1)
                G2.append(2*(values[:, [[i] for i in idxj], idxk].sum(axis=2).sum(axis=1)))
            else:    
                G2.append(values[:, [[i] for i in idxj], idxk].sum(axis=2).sum(axis=1))
            
        return G2, values
    

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
        dRij_dRml : array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                    Derivative of the norm of the position vector R_{ij}.
        Rij_dRij_dRml : array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                        R_{ij} multiplied by its derivative.
        
        """
        
        N_unit = self._N_unitcell
        
        ## construct the identity matrices first with all atoms in the i and m dimensions
        ## then pick out the relevant slices in the i and m dimensions which correspond to "central" atoms
        deltajm_deltaim = (numpy.eye(len(coords))[None,:,:N_unit]
                            - numpy.eye(len(coords))[:,None,:N_unit])

        Rij_dRij_dRml = deltajm_deltaim[:,:,:,None] * (coords[None,:,None,:] - coords[:,None,None,:])

        with numpy.errstate(divide='ignore', invalid='ignore'):
            dRij_dRml = Rij_dRij_dRml / R[:,:,None,None]     
        dRij_dRml = numpy.nan_to_num(dRij_dRml)
        
        return dRij_dRml, Rij_dRij_dRml


    def Rij_dRij_dRml_sum(self, Rij_dRij_dRml):

        """
        Sum of Rij.(dRij/dRml) terms:
    
        Parameters
        ----------
        Rij_dRij_dRml : array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                        R_{ij} multiplied by its derivative.
                        
        Returns
        -------
        array, shape=(N_atoms, N_atoms, N_atoms, N_centralatoms, 3)
        Sum of Rij.(dRij/dRml) terms.
        
        """
        
        N_unit = self._N_unitcell
            
        return (Rij_dRij_dRml[:N_unit,:,None,:,:] 
                + Rij_dRij_dRml[:N_unit,None,:,:,:] 
                + Rij_dRij_dRml[None,:,:,:,:])


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
        dfc : array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
              Derivative of the cutoff function. 

        """
        
        dfc = 0.5 * (-numpy.pi / self.r_cut) * numpy.sin(numpy.pi * R / self.r_cut)[:,:,None,None] * dRij_dRml
        dfc[R > self.r_cut] = 0
        dfc[R < 1E-08] = 0  ## in order to exclude self terms from sum 
        
        return dfc
    
    
    def fcinv_dfc_dRml_sum(self, fc, dfc_dRml):

        """
        Sum of (1/fc).(dfc/dRml) terms:
    
            
        Parameters
        ----------
        fc : array, shape=(N_atoms, N_atoms)
             The new distance matrix with the cutoff function applied
        dfc_dRml : array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   Derivative of the cutoff function. 
    
        Returns
        -------
        array, shape=(N_atoms, N_atoms, N_atoms, N_centralatoms, 3)
        Sum of (1/fc).(dfc/dRml) terms.

        """
        
        N_unit = self._N_unitcell
        
        with numpy.errstate(divide='ignore', invalid='ignore'):
            fcinv_dfc_dRml = dfc_dRml / fc[:,:,None,None]
        fcinv_dfc_dRml = numpy.nan_to_num(fcinv_dfc_dRml) 
        
        return (fcinv_dfc_dRml[:N_unit,:,None,:,:]
                + fcinv_dfc_dRml[:N_unit,None,:,:,:]
                + fcinv_dfc_dRml[None,:,:,:,:])


    def dcosTheta_dRml(self, coords, R, dRij_dRml, cosTheta, periodic=False): 
        
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
        cosTheta: array, shape=(N_centralatoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        dcosTheta : array, shape=(N_centralatoms, N_atoms, N_atoms, N_centralatoms, 3)
                         The derivatives of the cosines of triplet angles.
        
        """
        
        N_tot = len(coords)
        N_unit = self._N_unitcell
        
        R_vecs = coords - coords[:N_unit,None]

        ## construct the identity matrices first with all atoms in the i and m dimensions
        ## then pick out the relevant slices in the i and m dimensions which correspond to "central" atoms
        deltajm_deltaim = numpy.subtract(numpy.eye(N_tot)[None,:,:N_unit],
                                         numpy.eye(N_tot)[:N_unit,None,:N_unit])

        dcosTheta = numpy.add(deltajm_deltaim[:,:,None,:,None] * R_vecs[:,None,:,None,:],
                              deltajm_deltaim[:,None,:,:,None] * R_vecs[:,:,None,None,:])

        with numpy.errstate(divide='ignore', invalid='ignore'):
            dcosTheta = (dcosTheta / (R[:N_unit,:,None,None,None]*R[:N_unit,None,:,None,None])
                         - (dRij_dRml[:N_unit,:,None,:,:] / R[:N_unit,:,None,None,None]
                             + dRij_dRml[:N_unit,None,:,:,:] / R[:N_unit,None,:,None,None])
                         * cosTheta[:,:,:,None,None])
        dcosTheta = numpy.nan_to_num(dcosTheta)

        return dcosTheta
    

    def dG1_dRml(self, R, dRij_dRml, fc, dfc_dRml, periodic=False):
        
        """
        Derivative of G1 symmetry function.
        See Eq. 13b of the supplementary information of Khorshidi, Peterson, CPC(2016).

        Parameters
        ----------
        R : array, shape=(N_atoms, N_atoms)
            A distance matrix for all the atoms (scipy.spatial.cdist).
        dRij_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                   An array of the derivatives of the distance matrix
        fc: array, shape=(N_atoms, N_atoms)
            The new distance matrix with the cutoff function applied
        dfc_dRml: array, shape=(N_atoms, N_atoms, N_centralatoms, 3)
                  An array of the derivatives of fc.
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        dG1 : array, shape=(N_centralatoms, N_centralatoms, 3, N_elements)
              The atom-wise dG^1 evaluations.
            
        """
        
        N_unit = self._N_unitcell
        
        values = ((dRij_dRml[:N_unit,:,:,:] * (-2*self.eta * (R[:N_unit,:] - self.r_s) * fc[:N_unit,:])[:,:,None,None]
                    + dfc_dRml[:N_unit,:,:,:])
                    * numpy.exp(-self.eta * (R[:N_unit,:] - self.r_s) ** 2)[:,:,None,None])   
        
        dG1 = []
        for ele in self._elements_unique:
            ## find the positions of all atoms of type "ele"
            idxs = numpy.where(self._elements == ele)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            dG1.append(values[:,idxs,:,:].sum(axis=1))
        
        return dG1


    def dG2_dRml(self, Rij_dRij_dRml_sum, fcinv_dfc_dRml_sum, cosTheta, dcosTheta_dRml, G2vals, periodic=False):
        
        """
        Derivative of G2 symmetry function.
        See Eq. 13d of the supplementary information of Khorshidi, Peterson, CPC(2016)

        Parameters
        ----------
        Rij_dRij_dRml_sum: 
        fcinv_dfc_dRml_sum:
        cosTheta: array, shape=(N_centralatoms, N_atoms, N_atoms)
                  An array of cosines of triplet angles.
        dcosTheta_dRml: array, shape=(N_centralatoms, N_atoms, N_atoms, N_centralatoms, 3)
                        The derivatives of the cosines of triplet angles.
        G2vals: array, shape=(N_centralatoms, N_atoms, N_atoms)
                The angular terms inside the sum.
        periodic: boolean (False = cluster/molecule, True = 3D periodic structure)

        Returns
        -------
        dG2: array, shape=(N_centralatoms, N_centralatoms, 3, N_elementpairs)
             The atom-wise dG^2 evaluations.
            
        """
      
        with numpy.errstate(divide='ignore', invalid='ignore'):
            values = 1. / (1 + self.lambda_ * cosTheta)    
        values = numpy.nan_to_num(values)[:,:,:,None,None] * dcosTheta_dRml

        values = G2vals[:,:,:,None,None] * (self.lambda_ * self.zeta * values - 2*self.eta * Rij_dRij_dRml_sum + fcinv_dfc_dRml_sum)
                   
        dG2 = []
        for [ele1,ele2] in self._element_pairs:
            ## find the positions of all pairs of atoms of type (ele1,ele2)
            idxj,idxk = numpy.where(self._elements==ele1)[0],numpy.where(self._elements==ele2)[0]
            ## and sum over them
            ## each row corresponds to each atom in the structure
            
            if ele1 != ele2:
                ## double the sum over pairs of (ele1,ele2) and (ele2,ele1)
                dG2.append(2*(values[:, [[i] for i in idxj], idxk].sum(axis=2).sum(axis=1)))
            else:    
                dG2.append(values[:, [[i] for i in idxj], idxk].sum(axis=2).sum(axis=1))

        return dG2
    
    