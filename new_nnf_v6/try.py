import numpy as np
import math

G_weights_sum = []

final_output = []

hidden_neuron_number = 3

input_number = 2

first_bias = [1,1,1]

second_bias = [1]

second_weights = [2,2,2]

G_per_atom =np.array([1,2,3,4,5])

first_weights = np.array([[1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3], [1, 2, 3]])

for i in range(hidden_neuron_number):
      print(first_weights[:,i])
      G_weights_sum.append(np.sum(np.multiply(G_per_atom, np.array(first_weights[:,i]))))

print(G_weights_sum)

for i in range(0, hidden_neuron_number):

      final_output.append(second_weights[i] * (1 / (1 + math.exp(-(G_weights_sum[i] + first_bias[i])))))

sum_of_all_terms = np.sum(final_output)

energy_per_atom = sum_of_all_terms + second_bias

print(G_weights_sum)
print(final_output)
print(sum_of_all_terms)
print(energy_per_atom)
