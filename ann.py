import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model
import numpy as np
#import pandas as pd
from molml.atom import BehlerParrinello
from ase.calculators.calculator import Calculator, all_changes
from sklearn import datasets, preprocessing
from itertools import combinations
from scipy.spatial.distance import cdist
import copy

BP_DEFAULT = [6.0,1.0,1.0,1.0,1.0]

#TODO: eliminate data loading from disk
def load_data(data_file='train_data.csv'):
    f = open(data_file)
    inputs = []
    for line in f.readlines():
        input = [float(x) for x in line.strip().split(',')]
        temp = []
        for i in range(0, int(len(input)/2)*2, 2):
            temp.append(input[i:i + 2])
        inputs.append(temp)
    return np.array(inputs)

class ATOMModel:
    def __init__(self, restore='atom_model', n=55):
        """
        Parameters
        ----------
        restore: weights obstained from data training
        atoms: Atoms object containing structure information to be predicted
        n: number of parrallel neural network. in the current method, it is number of atoms
        """
        #self.atoms = atoms
        self.n = n
        
        #contruct neural network model using weights from 'restore'
        inputs_data = []
        for i in range(n):
            inputs_data.append(Input((2,), ))

        atom_input = Input((2,))
        hidden = Dense(3, activation='sigmoid')(atom_input)
        prediction = Dense(1)(hidden)
        model = Model(inputs=atom_input, outputs=prediction)

        outputs = []
        for input in inputs_data:
            outputs.append(model(input))

        result = keras.layers.add(outputs)

        final_model = Model(inputs_data, result)

        final_model.load_weights(restore)
        self.prediction = final_model

    def predict(self, atoms):
        #Convert new structure to symmetry function and do normalization
        inputs = []
        predict_data = [self.to_symmetry_function(atoms=atoms)]
        normalized_data = np.reshape(self.normalize_data(predict_data),(self.n,2))
        for i in range(self.n):
            inputs.append(np.array([normalized_data[i]]))
        """
        Model accepts the following data structure: [array([[ g1_0,g2_0]]),array([[ g1_1,g2_1]]...array([[g1_n, g2_n]])]
        """
        return self.prediction.predict(inputs)

    #TODO:normalize new data based on parameters from training process instead of stacking training data and new data every time
    def normalize_data(self,data):
        train_data = load_data('train_data.csv')                     #load the trained data
        numb_stru = len(train_data)+len(data)
        all_data = np.row_stack((train_data,data))           #stack the trained and predicted#
        all_data2 = all_data.reshape(numb_stru,2*self.n)                        
        normalized = preprocessing.minmax_scale(all_data2)           #Max-Min normalization of the stacked data#
        return normalized[-1,:]                             #Grab the last line of the normalized data# 
        
    def to_symmetry_function(self, atoms=None, parameters=BP_DEFAULT):
        
        """
        
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
              
        Returns
        -------
        array of [[g1_0, g2_0], [g1_1, g2_1], ... , [g1_n, g2_n, E]] for n=number of atoms
        
        Notes
        -----
        Behler-Parrinello symmetry functions as described in:
        Behler, J; Parrinello, M. Generalized Neural-Network Representation of
        High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401
        Using the implementation in molML (https://pypi.python.org/pypi/molml/0.6.0)
        ** correct the angular term in the g_2 function!! **
        
        """
        if atoms is None:
           print "Atoms object is not given"
           sys.exit()

        #fetch coordinates and elements from atoms
        coords = atoms.get_positions()
        elements = atoms.get_chemical_symbols()
           
        r_cut, r_s, eta, lambda_, zeta = parameters
    
        bp = BehlerParrinello(r_cut=r_cut, r_s=r_s, eta=eta,
                              lambda_=lambda_, zeta=zeta)
        bp._elements = elements
        bp._element_pairs = set(combinations(elements,2))
        g_1 = bp.g_1(cdist(coords, coords), elements = elements)[:,0]
        g_2 = bp.g_2(Theta = bp.calculate_Theta(R_vecs = coords), 
                     R = cdist(coords, coords), elements = elements)
    
        return np.column_stack((g_1,g_2))

class Ann(Calculator):
    """
    Calculator using artificial neural network to predict energy and force
    """

    implemented_properties = ['energy', 'forces']

    default_parameters = dict(
         stepsize = 0.02)

    def __init__(self, restart=None, ignore_bad_restart_file=False, 
                 label=None, atoms=None, ann_model=None, **kwargs):
        self.ann_model = ann_model
        self.label = None
        self.atoms = None
        self.parameters = None
        self.results = None
        
        if self.ann_model is None:
           print "please assign ANN model"
           sys.exit()

        Calculator.__init__(self, restart, ignore_bad_restart_file, label, atoms, **kwargs)

    def set(self, **kwargs):
        """Set parameters like set(key1=value1, key2=value2,
        ...)."""
        changed_parameters = Calculator.set(self, **kwargs)
        if changed_parameters:
           self.reset()

    #run calculation to get energy and forces, which will be stored in dictionary 'results'
    def calculate(self, atoms=None, properties=None, system_changes=all_changes):
        """calculate and store energy"""
        if not properties:
           properties = 'energy'

        """
        If system_changes, atoms etc will be updated.
        """
        Calculator.calculate(self, atoms, properties, system_changes)

        e_old = self.ann_model.predict(self.atoms)[0][0]

        self.results['energy'] = float(e_old)
        self.results['forces'] = np.array(self.cal_force(e_old))
       
    def cal_force(self, energy):
        forces = []
        stepsize = self.parameters.stepsize
        e_old = energy
        #TODO: using MPI for parallelization
        for atom in self.atoms:
            force = []
            print "Perturbation on atom ",atom.index
            for i in range(3):
                temp_position = copy.copy(atom.position)
                atom.position[i] += stepsize
                e_new = float(self.ann_model.predict(self.atoms)[0][0])
                force.append(-(e_new - e_old)/stepsize)
                atom.position = copy.copy(temp_position)
            forces.append(np.array(force))
        return forces
