import numpy as np
import math


###############################

Read the weights and bias from *.csv files

###############################

first_weights = np.loadtxt(open("first_weights.csv","rb"), delimiter=",",skiprows=0)
second_weights = np.loadtxt(open("second_weights.csv","rb"), delimiter=",",skiprows=0)
first_bias = np.loadtxt(open("first_bias.csv","rb"), delimiter=",",skiprows=0)
second_bias = np.loadtxt(open("second_bias.csv","rb"), delimiter=",",skiprows=0)

hidden_neuron_number = len(np.array(first_weights[0]))  #number of hidden neurons#
input_number = len(first_weights)                       #number of inputs#

G_weights_sum = []
subnetwork_final_output = []

print(first_weights,input_number, hidden_neuron_number)

for i in range(hidden_neuron_number):
      print(first_weights[:,i])
      G_weights_sum.append(np.sum(np.multiply(G_per_atom, np.array(first_weights[:,i]))))                    # sum all the G*w  #

for i in range(0, hidden_neuron_number):

      final_output.append(second_weights[i] * (1 / (1 + math.exp(-(G_weights_sum[i] + first_bias[i])))))     #sigmoid activation function#

sum_of_all_terms = np.sum(subnetwork_final_output)                  #sum all the terms in a subnetwork#

sub_network_energy_per_atom = sum_of_all_terms + second_bias        #calculate the energy per atom of a subnetwork#
