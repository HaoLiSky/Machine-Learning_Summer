import numpy as np

G = [1, 2]

input_number = 2

neuron_number = 3

weights = [[2, 2, 2],
           [3, 3, 3]]

bias1 = [4, 4, 4]

print(weights)

Gs = []

G_t_W = []



G_t_W.append(G[0] * weights[0][0] + G[1] * weights[0][0] + bias1[0])

G_t_W.append(G[0] * weights[0][1] + G[1] * weights[1][1] + bias1[1])

G_t_W.append(G[0] * weights[0][2] + G[1] * weights[1][2] + bias1[2])

print(G_t_W) 


G_t_W2 = []
G_t_W3 = []
b = 0


for i in range(0, 1):

    c = G[i] * weights[0][0]
    b = b + c

#    G_t_W2.append(G[0] * weights[0][1] + G[1] * weights[1][1] + bias1[1])

#   G_t_W2.append(G[0] * weights[0][2] + G[1] * weights[1][2] + bias1[2])

print(b)
