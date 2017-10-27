import sys
import numpy as np
from molml.atom import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations

'''
usage:
>>>python representation.py data.npz 1500 output.csv [convert raw_data.xyz 55]

Behler, J; Parrinello, M. Generalized Neural-Network Representation of
High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401.
r_cut=6.0, cutoff radius (angstroms)
r_s=1      pairwise distance offset (angstroms)
eta=1      exponent term dampener
lambda_=1  angular expansion switch +1 or -1
zeta=1     angular expansion degree
'''

BP_DEFAULT = [6.0, 1, 1, 1, 1]


def convert_xyz(inputfile, outputfile, splitter, ignore=['energy:']):
    with open(inputfile, 'r') as fil:
        text = fil.read()
        chunks = text.split('\n'+splitter+'\n')

    energies = [x.split()[0] for x in text.split('energy:')[1:]]

    data = [list(filter(lambda x: (x and not x == splitter
                                   and all([x.find(term) == -1
                                           for term in ignore])),
                        x.split('\n'))) for x in chunks if x]

    structures = sorted([[y.split() for y in x] for x in data],
                        key=lambda struct: struct[0])

    # sorted by atom type
    np.savez_compressed(outputfile, structures=structures, energies=energies)


def parser(structures, nmax):
    i = 0
    structure = structures[i]
    species = [atom[0] for atom in structure]
    coords = [[float(coord) for coord in atom[1:]] for atom in structure]
    conversion = (species, coords)

    while i < nmax:
        yield conversion
        structure = structures[i]
        species = [atom[0] for atom in structure]
        coords = [[float(coord) for coord in atom[1:]] for atom in structure]
        conversion = (species, coords)
        i += 1

def represent(coords, elements, energy, parameters=BP_DEFAULT):
    '''
    coords := a list of coordinate triplets
    elements := a list of strings
    '''
    r_cut, r_s, eta, lambda_, zeta = parameters

    bp = BehlerParrinello(r_cut=r_cut, r_s=r_s, eta=eta,
                          lambda_=lambda_, zeta=zeta)
    bp._elements = elements
    bp._element_pairs = set(combinations(elements,2))
    g_1 = bp.g_1(cdist(coords, coords), elements = elements)[:,0]
    g_2 = bp.g_2(Theta = bp.calculate_Theta(R_vecs = coords), 
                 R = cdist(coords, coords), elements = elements)

    return np.append(np.ravel(np.column_stack((g_1,g_2))), energy)

    # format: [g1_0, g2_0, g1_1, g2_1, ... , g1_n, g2_n, E] for n=number of atoms

if __name__ == '__main__':
    arrayfile = sys.argv[1]
    outputfile = sys.argv[2]
    Nmax = sys.argv[3]

    if len(sys.argv) > 4 and sys.argv[4] == 'convert':
        inputfile = sys.argv[5]
        splitphrase = sys.argv[6]
        convert_xyz(inputfile, arrayfile, splitphrase)

    arraydata = np.load(arrayfile)
    structures = arraydata['structures']
    energies = arraydata['energies']

    N = min(len(structures),len(energies), int(Nmax))
    print(('Total structures: '+str(N)).ljust(50))

    XYZ = parser(structures, N)
    j = 0
    with open(outputfile,'w') as fil:
        for (elements, coords) in XYZ:
            print('processing '+str(j).rjust(10)+'/'+str(N),end='\r')
            sys.stdout.flush()
            try:
                energy = energies[j]
                data = represent(np.asarray(coords), elements, energy)
                line = ','.join(data)+'\n'
                fil.write(line)
                j += 1
            except:
                continue
