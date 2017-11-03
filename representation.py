import sys, argparse, time
import numpy as np
#from molml.atom import BehlerParrinello
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations


BP_DEFAULT = [6.0,1.0,1.0,1.0,1.0]


def convert_xyz(xyzinputfile, splitter, ignore=['energy:']):
    
    """
    Read in information from xyz file and store it in a numpy compressed zip file
    
    Parameters
    ----------
    xyzinputfile : xyz data filename
    splitter : string used to distinguish between adjacent structure data in the xyz file
               (the line that states the number of atoms?)
          
    Returns
    -------
    structures : list of structure data for each structure in xyz file
    energies : list of energies corresponding to each structure
    
    """ 
    
    with open(xyzinputfile, 'r') as fil:
        text = fil.read()
        chunks = text.split('\n'+splitter+'\n')

    energies = [x.split()[0] for x in text.split('energy:')[1:]]

    data = [list(filter(lambda x: (x and not x == splitter and all([x.find(term) == -1 for term in ignore])),
                        x.split('\n'))) for x in chunks if x]
    
    structures = [[y.split() for y in x] for x in data]

    return structures, energies


def parser(structures, Nmax):
    
    """
    
    Parameters
    ----------
    structures : list of structure data for each structure in xyz file
    Nmax: maximum number of structures to be included in the representation portion\'s output
          
    Returns
    -------
    generator object (species,coords)
    
    """ 
    
    i = 0
    while i < Nmax:
        structure = structures[i]
        species = [atom[0] for atom in structure]
        coords = [[float(coord) for coord in atom[1:]] for atom in structure]
        yield (species, coords)
        i += 1


def represent(coords, elements, energy, parameters=BP_DEFAULT):
    
    """
    
    Parameters
    ----------
    coords: list of [xyz] coords (in Angstroms)
    elements: list of element name strings
    energy: total energy of structure (in eV)
    parameters: list of parameters for Behler-Parrinello symmetry functions
        r_cut: cutoff radius (angstroms); default = 6.0
        r_s: pairwise distance offset (angstroms); default = 1.0
        eta: exponent term dampener; default = 1.0
        lambda_: angular expansion switch +1 or -1; default = 1.0
        zeta: angular expansion degree; default = 1.0
          
    Returns
    -------
    array of [g1_0, g2_0, g1_1, g2_1, ... , g1_n, g2_n, E] for n=number of atoms
    
    Notes
    -----
    Behler-Parrinello symmetry functions as described in:
    Behler, J; Parrinello, M. Generalized Neural-Network Representation of
    High-Dimensional Potential-Energy Surfaces. Phys. Rev. Lett. 98, 146401
    Using the implementation in molML (https://pypi.python.org/pypi/molml/0.6.0)
    ** correct the angular term in the g_2 function!! **
    
    """ 
    
    r_cut, r_s, eta, lambda_, zeta = parameters

    bp = BehlerParrinello(r_cut=r_cut, r_s=r_s, eta=eta, lambda_=lambda_, zeta=zeta)
    bp._elements = elements
    bp._element_pairs = set(combinations(elements,2))
    
    ## we do not have to recompute the distances and angles for every different set of parameters
    distmatrix = cdist(coords, coords)
    cosTheta = bp.calculate_cosTheta(coords, distmatrix)
    
    ## here we can add loops over different sets of parameters, overwriting the default values that were used to initialize the class
#    bp.r_cut,bp.r_s,bp.eta,bp.lambda_,bp.zeta = [6.0],[1.0],[1.0],[1.0],[1.0]
#    for ...
    g_1 = bp.g_1(R = distmatrix, elements = elements)[:,0]
    g_2 = bp.g_2(cosTheta = cosTheta, R = distmatrix, elements = elements)    

    return np.append(np.ravel(np.column_stack((g_1,g_2))), energy)


if __name__ == '__main__':
    
    
    argparser = argparse.ArgumentParser(description='Converts structures to symmetry functions.')
    argparser.add_argument('npz',
                        help='filename for the numpy compressed zip produced by the convert option')
    argparser.add_argument('Nmax',type=int,
                        help='maximum number of structures to be included in the representation portion\'s output')
    argparser.add_argument('output',
                        help='filename for the representation portion\'s csv output (i.e. the neural network input)')
    argparser.add_argument('-convert',action="store_true",
                        help='keyword argument to start data conversion, which must be run if data.npz does not exist')
    argparser.add_argument('-rawdata',
                        help='filename for the original data in xyz format')
    argparser.add_argument('-separator',
                        help='string used to distinguish between adjacent structure data in the xyz file')    
   
    ## read in the arguments from the command line
    args = argparser.parse_args()
    arrayfile = args.npz 
    Nmax = args.Nmax
    outputfile = args.output

    ## read in data from the xyz file and store it in a numpy compressed zip file
    if len(sys.argv) > 4 and args.convert:
        xyzinputfile = args.rawdata
        splitphrase = args.separator
        structures,energies = convert_xyz(xyzinputfile, splitphrase)
        np.savez_compressed(arrayfile, structures=structures, energies=energies)
    
    ## extract arrays from the numpy compressed zip fileim[ort]
    arraydata = np.load(arrayfile)
    structures = arraydata['structures']
    energies = arraydata['energies']

    N = min(len(structures),len(energies), int(Nmax))
    print(('Total structures: '+str(N)).ljust(50))
    
    ## convert structure coordinates into symmetry functions
    XYZ = parser(structures, N)
    
    t0 = time.time()
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
    print (time.time()-t0)
            
