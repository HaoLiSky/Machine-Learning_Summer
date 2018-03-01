import keras
from keras.models import Sequential
from keras.layers import Dense, Activation, Input
from keras.models import Model
import numpy as np
import keras.backend as K
#import pandas as pd
import math
from nnf.io_utils import load_nn_paras
from nnf.batch_fingerprint import parameters_from_file
from nnf.fingerprints import represent_BP
from ase.calculators.calculator import Calculator, all_changes
from itertools import combinations
from scipy.spatial.distance import cdist
import copy

BP_DEFAULT = [6.0,1.0,1.0,1.0,1.0]

"""
NN_model class: fetch parameters (i.e., weights, bias, etc.) from trained model and then
reconstruct neural network model for energy and force prediction
"""
class NN_Model:
    def __init__(self, normalize_paras_file=None, nn_paras_file='nn_paras.csv',act_function = 'sigmoid',
                 n_hidden_layer = 2, parameters_file=None, periodic=False, derivs=False):
        """
        Initialize NN_model:
        normalize_paras: max and min values used for normalization
            format:
             array([[max_1,...,max_n],[min_1,...,min_n]])
        nn_paras: weights and bias obtained from neural network training.
            format: dictionary with layer number as key and (n_neuron_prev * n_neuron_curr) array as value
             n_neural_curr: number of neurons in the current layer 
             n_neural_prev: number of neurons in the previous layer 
             {"layer_1":array([[],[],...,[]];...;"layer_l":array([[],[],...,[]])}
        Parameters
        ----------
        """
        self.parameters = parameters_from_file(parameters_file)
        self.sys_elements = None
        self.periodic = periodic
        self.derivs = derivs
        self.act_function = act_function
        #load nn model parameters (weights, bias, etc) and normalize parameters([min],[max],[span])
        self.normalize_paras=np.loadtxt(open(normalize_paras_file,'r'),delimiter=',',skiprows=0)
        self.nn_paras = load_nn_paras(nn_paras_file)
        #print(self.normalize_paras)
        #TODO: store span in nn_paras_file in advance
        self.span = self.normalize_paras[1] - self.normalize_paras[0]
        if derivs:
           self.derivatives = []
        self.n_hidden_layer = n_hidden_layer
        if self.n_hidden_layer is None:
           for key in self.nn_paras:
             if 'layer' in key:
                self.n_hidden_layer+=1

    #TODO: add more activation function
    def activation_function(self, x):
        if self.act_function == 'sigmoid':
           sigmoid_f = (1/(1+math.exp(-x)))-0.5
           if self.derivs:
              sigmoid_df = (sigmoid_f+0.5) * (1 - (sigmoid_f)+0.5)
              return sigmoid_f, sigmoid_df
           return sigmoid_f, None
          # return (K.sigmoid(x))-0.5,None

    #calculate energy for each atom
    def energy_per_atom(self, G_per_atom):
        derivative_per_atom = []
        for i in range(self.n_hidden_layer):
           #input layer as corner situation
           if i == 0:
              prev_layer = G_per_atom
              derivs_prev_layer = 1.0
           curr_layer = []
           derivative_per_layer = []
           curr_weights = self.nn_paras['layer_'+str(i+1)]
           print("    layer",i)
           for j in range(len(curr_weights[0])):
              #get inputs for each neuron for current hidden layer. TODO:add bias if needed
              #curr_layer.append(self.activation_function(np.sum(np.multiply(prev_layer, np.array(curr_weights[:,j]))+curr_bias)))
              weight = np.array(curr_weights[:,j]) 
              #sigmoid_f, sigmoid_df = self.activation_function(np.sum(np.multiply(prev_layer, weight)))
              sigmoid_f, sigmoid_df = self.activation_function(np.dot(prev_layer, weight))
              curr_layer.append(sigmoid_f)
              #df/dg for each neuron: array([df_1/dg_1, df_1/dg_2,...,df1/dg_ng]). two-dimensional (#ofNeurons * #ofG_functions)
              if i == 0:
                 derivative_per_layer.append(sigmoid_df * weight)
                 continue
              #Chain rule of derivative
              derivative_per_layer.append(np.sum(np.matmul(sigmoid_df * weight, np.array(derivs_prev_layer))))
           print("    ",curr_layer)
           prev_layer = np.array(curr_layer)
           derivs_prev_layer = derivative_per_layer
        end_weights = np.array(self.nn_paras['layer_'+str(self.n_hidden_layer+1)][:,0])
        
        return np.dot(prev_layer,end_weights), np.matmul(end_weights, derivs_prev_layer)
        #return np.sum(np.multiply(prev_layer,np.array(self.nn_paras['layer_'+str(self.n_hidden_layer+1)][:,0]))

    def predict_energy(self,atoms): 
        """
        g_list: a (N_atom * n_g_function) array
          array([g_1,g_2, ..., g_n]) where g_n = [g(1),g(2),...,g(n_g_function)]
        """
        #print(self.nn_paras['layer_'+str(self.n_hidden_layer+1)][:,0])
        self.to_symmetry_function(atoms)

        #get # of symmetry functions
        n_g = len(self.g_dg_list[0])

        g_list = np.column_stack((self.g_dg_list[0].reshape(n_g,-1),self.g_dg_list[1].reshape(n_g,-1)))
        energy = 0
        for i in range(len(g_list)):
           print("atom",i)
           #normalization
           #if i ==0:
           #   print("before n")
           #   print(g_list[i])
           #print(len(self.normalize_paras[0]),len(self.span))
           g = (np.array(g_list[i]) - self.normalize_paras[0]) /self.span
           #if i==0:
           #   print("after n")
           #   print(g)
           e_per_atom, derivs_per_atom = self.energy_per_atom(g) 
           energy += e_per_atom
           self.derivatives.append(derivs_per_atom)
        self.derivatives = np.array(self.derivatives)
        #print(self.derivatives)
        #print(energy)
        return energy

    def predict_force(self, atoms):
        energy = self.predict_energy(atoms)
        dg_list = [self.g_dg_list[2],self.g_dg_list[3]]
        forces = []
        #print(self.derivatives.shape)
        #print(np.squeeze(np.dstack(((np.array(dg_list[0][0])[:,0]),np.array(dg_list[1][0])[:,0])),axis=1))
        #print(np.squeeze(np.dstack(((np.array(dg_list[0][0])[:,0]),np.array(dg_list[1][0])[:,0])),axis=1).shape)
        for i in range(len(atoms)):
          de_dx = np.sum(np.matmul(self.derivatives, np.transpose(np.squeeze(np.dstack(((np.array(dg_list[0][i])[:,0]),np.array(dg_list[1][i])[:,0])),axis=1))))
          de_dy = np.sum(np.matmul(self.derivatives, np.transpose(np.squeeze(np.dstack(((np.array(dg_list[0][i])[:,1]),np.array(dg_list[1][i])[:,1])),axis=1))))
          de_dz = np.sum(np.matmul(self.derivatives, np.transpose(np.squeeze(np.dstack(((np.array(dg_list[0][i])[:,2]),np.array(dg_list[1][i])[:,2])),axis=1))))
          forces.append(np.array([de_dx,de_dy,de_dz]))
        return np.array(forces)

    def to_symmetry_function(self, atoms=None):
        
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
        """
        if atoms is None:
           print("Atoms object is not given")
           sys.exit()
        n_atoms = len(atoms)
        #fetch coordinates and elements from atoms
        coords = np.array(atoms.get_positions())
        elements = np.array(atoms.get_chemical_symbols())
        if self.sys_elements is None:
           sys_elements = elements
        else:
           sys_elements = self.sys_elements
        elements = np.array(atoms.get_chemical_symbols())
        """
        g_list format: [array([[[g1_atom_1]], [[g1_atom_2]], ..., [[g1_atom_n]]],
                        array([[[g2_atom_1]], [[g2_atom_2]], ..., [[g2_atom_n]]] ]
        """
        #g_orders used for binary system TODO
        self.g_dg_list, g_orders = represent_BP(coords = coords, elements = elements, sys_elements = sys_elements,
                                       parameters = self.parameters, derivs = self.derivs, periodic= self.periodic, 
                                       N_unitcell = n_atoms)
        #print("g_dg")
        #print(self.g_dg_list)
        return self.g_dg_list

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
           print("please assign ANN model")
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
            print("Perturbation on atom ",atom.index)
            for i in range(3):
                temp_position = copy.copy(atom.position)
                atom.position[i] += stepsize
                e_new = float(self.ann_model.predict(self.atoms)[0][0])
                force.append((e_new - e_old)/stepsize)
                atom.position = copy.copy(temp_position)
            forces.append(np.array(force))
        return forces
