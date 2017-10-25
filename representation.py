import sys
import numpy as np
from molml.atom import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations

# Behler, J; Parrinello, M. Generalized Neural-Network Representation of
#     High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.

def parser(filename, max, struct_split='55\n',ignore=['energy']):

    with open(filename,'r') as fil:
        structures = fil.read().split(struct_split)

    structures = [structure for structure in structures if structure]
    i = 0

    lines = list(filter(lambda x: any([(x.find(y) == -1) for y in ignore]),
                        structures[i].split('\n')))

    conversion = [line.split() for line in lines]

    while (i < max):
        yield conversion
        lines = list(filter(lambda x: any([(x.find(y) == -1) for y in ignore]),
                            structures[i].split('\n')))
        conversion = [line.split() for line in lines]
        i += 1

def load_energies(file_name):
    with open(file_name,'r') as fil:
        data = [line for line in fil.readlines() if line.find('energy') > -1]
        energies = [line.split()[3] for line in data]

    return energies


def represent(coords, elements, energy):
    '''
    coords := a list of coordinate triplets
    elements := a list of strings
    '''
    bp = BehlerParrinello()
    bp._elements = elements
    bp._element_pairs = set(combinations(elements,2))
    g_1 = bp.g_1(cdist(coords, coords), elements = elements)[:,0]
    g_2 = bp.g_2(Theta = bp.calculate_Theta(R_vecs = coords), 
                 R = cdist(coords, coords), elements = elements)
    #plt.hist(g_1, bins=20, log=True);
    #plt.hist(g_2, bins=20, log=True);
    #return np.ravel(np.column_stack((g_1,g_2))
    return np.append(np.ravel(np.column_stack((g_1,g_2))), energy)

    #format: [g1_0, g2_0, g1_1, g2_1, ... , g1_n, g2_n, E] for n=number of atoms

if __name__ == '__main__':
    file_name = sys.argv[1]
    Nmax = sys.argv[2]
    
    energies = load_energies(file_name)

    N = min(len(energies), int(Nmax))
    print(('Total structures: '+str(N)).ljust(50))

    XYZ = parser('new_au55_all.xyz', N)
    lines = []
    j = 0

    with open('output','w') as fil:
        for structure in XYZ:
            print('processing '+str(j).rjust(10)+'/'+str(N),end='\r')
            sys.stdout.flush()
            elements = [atom[0] for atom in structure if atom]
            coords = [[float(k) for k in atom[1:]] for atom in structure if atom]
            energy = energies[j]; j += 1
            try:
                data = represent(np.asarray(coords), elements, energy)
                line = ','.join(data)+'\n'
                fil.write(line)
                #lines.append(line+'\n')
            except:
                continue
#    with open('output','w') as fil:
 #       fil.writelines(lines)
