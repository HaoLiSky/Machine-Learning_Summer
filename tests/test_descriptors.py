import unittest
import numpy as np
from scipy.misc import comb
from ase.build import bulk
from fingerprints import represent_BP, build_supercell


class Test_Gs(unittest.TestCase):
    
    """
    test evaluation of symmetry functions G1 and G2.
    """   
    
    def test_G1shape(self):
        
        """
        """

        num_elements = len(np.unique(self.elements))
        
        self.assertEqual(self.G1.shape[0],self.N_unitcell)
        self.assertEqual(self.G1.shape[1],num_elements)
        self.assertEqual(self.G1.shape[2],len(self.para_pairs))
        
        
    def test_G2shape(self):
        
        """
        """

        num_elements = len(np.unique(self.elements))

        self.assertEqual(self.G2.shape[0],self.N_unitcell)
        self.assertEqual(self.G2.shape[1],(comb(num_elements,2)+num_elements))
        self.assertEqual(self.G2.shape[2],len(self.para_triplets))

        
    def test_G1values(self):
        
        """
        """

        if self.toy:
            for i in range(self.G1.shape[0]):
                for el in range(self.G1.shape[1]):
                    for para in range(self.G1.shape[2]):
                        self.assertAlmostEqual(self.G1[i,el,para],self.G1ref[i,el,para],places=12)            


    def test_G2values(self):
        
        """
        """

        if self.toy:
            for i in range(self.G2.shape[0]):
                for el in range(self.G2.shape[1]):
                    for para in range(self.G2.shape[2]):
                        self.assertAlmostEqual(self.G2[i,el,para],self.G2ref[i,el,para],places=12)
 
                        
class Test_dGs(unittest.TestCase):
    
    """
    test evaluation of derivatives of symmetry functions dG1 and dG2.
    """   

    def test_dG1shape(self):
        
        """
        """
        
        num_elements = len(np.unique(self.elements))
        
        self.assertEqual(self.dG1.shape[0],self.N_unitcell)
        self.assertEqual(self.dG1.shape[1],self.N_unitcell)
        self.assertEqual(self.dG1.shape[2],3)
        self.assertEqual(self.dG1.shape[3],num_elements)
        self.assertEqual(self.dG1.shape[4],len(self.para_pairs))
        
        
    def test_dG2shape(self):
        
        """
        """
        
        num_elements = len(np.unique(self.elements))

        self.assertEqual(self.dG2.shape[0],self.N_unitcell)
        self.assertEqual(self.dG2.shape[1],self.N_unitcell)
        self.assertEqual(self.dG2.shape[2],3)
        self.assertEqual(self.dG2.shape[3],(comb(num_elements,2)+num_elements))
        self.assertEqual(self.dG2.shape[4],len(self.para_triplets))
        
        
    def test_dG1_fd(self):
         
        """
        """
        
        disp = 1E-04
        for m in range(self.dG1.shape[1]):
            for l in range(3):
                self.coords[m][l] += disp
                G1d,G2d = represent_BP(np.asarray(self.coords), np.asarray(self.elements), self.system_elements,
                                       [self.para_pairs,self.para_triplets], 
                                       derivs=False, periodic=self.periodic, N_unitcell=self.N_unitcell)[0]
                self.coords[m][l] -= disp
                for i in range(self.dG1.shape[0]):
                    for el in range(self.dG1.shape[3]):
                        for para in range(self.dG1.shape[4]):
                            self.assertAlmostEqual(self.dG1[i,m,l,el,para], (G1d[i,el,para] - self.G1[i,el,para])/disp, delta=disp)


    def test_dG2_fd(self):
         
        """
        """
        
        disp = 1E-04
        for m in range(self.dG2.shape[1]):
            for l in range(3):
                self.coords[m][l] += disp
                G1d,G2d = represent_BP(np.asarray(self.coords), np.asarray(self.elements), self.system_elements,
                                       [self.para_pairs,self.para_triplets], 
                                       derivs=False, periodic=self.periodic, N_unitcell=self.N_unitcell)[0]
                self.coords[m][l] -= disp
                for i in range(self.dG2.shape[0]):
                    for el in range(self.dG2.shape[3]):
                        for para in range(self.dG2.shape[4]):
                            self.assertAlmostEqual(self.dG2[i,m,l,el,para], (G2d[i,el,para] - self.G2[i,el,para])/disp, delta=disp)


class toy_molecule(Test_Gs,Test_dGs):

    def setUp(self):
        
        self.coords = [np.array([-1.0,0.0,0.0]),np.array([0.0,0.0,0.0]),np.array([0.8,0.0,0.0])]
        self.elements = ['B','A','A']
        self.system_elements = ['B','A']
        self.N_unitcell = len(self.coords)
        self.periodic = False
        
        self.para_pairs = [[6.0, 0.0, 1.0, 1.0, 1.0]]
        self.para_triplets = [[6.0, 0.0, 1.0, 1.0, 1.0]]
        
        self.derivs = True
        fingerprints = represent_BP(np.asarray(self.coords), np.asarray(self.elements), self.system_elements,
                                    [self.para_pairs,self.para_triplets],
                                    derivs=self.derivs, 
                                    periodic=self.periodic, N_unitcell=self.N_unitcell)[0]
        self.G1,self.G2,self.dG1,self.dG2 = fingerprints
        
        self.toy = True
        self.G1ref = np.array([[[0.],[0.34323619137796718 + 0.031091927530250564]],
                               [[0.34323619137796718],[0.50449901143846576]],
                               [[0.031091927530250564],[0.50449901143846576]]])
        self.G2ref = np.array([[[0.],[0.],[0.010767900561565893 * 2]],
                               [[0.],[0.],[0.]],
                               [[0.],[0.010767900561565893 * 2],[0.]]])
        

class toy_periodic(Test_Gs,Test_dGs):

    @classmethod
    def setUpClass(cls):
        
        unitcell = bulk('Cu','fcc',a=4.078,cubic=True)
        cls.coords, cls.elements, cls.N_unitcell = build_supercell(unitcell, R_c = 6.0)
        cls.system_elements = ['Cu']
        cls.periodic = True
        
        cls.para_pairs = [[6.0, 0.0, 0.1, 1.0, 1.0]]
        cls.para_triplets = [[6.0, 0.0, 0.1, 1.0, 1.0]]

        cls.derivs = True
        fingerprints = represent_BP(np.asarray(cls.coords), np.asarray(cls.elements), cls.system_elements,
                                    [cls.para_pairs,cls.para_triplets],
                                    derivs=cls.derivs, 
                                    periodic=cls.periodic, N_unitcell=cls.N_unitcell)[0]
        cls.G1,cls.G2,cls.dG1,cls.dG2 = fingerprints

        cls.toy = True
        cls.G1ref = np.array([[[3.1717067072824454]],
                              [[3.1717067072824454]],
                              [[3.1717067072824454]],
                              [[3.1717067072824454]]])
        cls.G2ref = np.array([[[1.2109569172462602]],
                              [[1.2109569172462602]],
                              [[1.2109569172462602]],
                              [[1.2109569172462602]]])
        
        
if __name__ == '__main__':


    suite1 = unittest.TestLoader().loadTestsFromTestCase(toy_molecule) 
    unittest.TextTestRunner(verbosity=2).run(suite1)
    
    suite2 = unittest.TestLoader().loadTestsFromTestCase(toy_periodic) 
    unittest.TextTestRunner(verbosity=2).run(suite2)
    
