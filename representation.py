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


def represent(coords, elements, parameters=BP_DEFAULT):
    
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

    n = len(g_1)  #TEMP
    k = 2  #TEMP
    return np.asarray([g_1,g_2]).reshape(n,k)


def pad(data, sects_i, sects_f):
    # data = list of descriptors for one structure
    # sects_i = list of initial lengths of chunks
    #   i.e. # of atoms for each specie in structure
    # sects_f = list of final lengths of chunks
    #   i.e. max # of atoms for each specie in system
    # returns data padded with 0 matching the template

    columns = np.shape(data)[1]  # length of each descriptor
    rows_f = sum(sects_f)

    n = len(sects_i)  # number of sections
    slice_pos = [sum(sects_i[:i + 1]) for i in range(n - 1)]
    # row indices to slice to create n sections of sects_i length
    data_sliced = np.split(data, slice_pos)

    start_pos = [sum(sects_f[:i]) for i in range(n)]
    # row indices to place chunks of sects_f length

    data_f = np.zeros((rows_f, columns))
    for sect, start in zip(data_sliced, start_pos):
        end = start + len(sect)
        data_f[start:end, :] = sect

    return data_f


if __name__ == '__main__':


    """
    Usage:
    [parse|convert] input output [-n nmax] [-istart index]
    [-e energy_file]  [-ignore ...]
    """
    
    argparser = argparse.ArgumentParser(description='Converts structures to symmetry functions.')
    argparser.add_argument('action', choices=['represent','parse'])
    argparser.add_argument('input',
                           help='.npz for representation or .xyz for \
                                parsing (directory if -loose)')
    argparser.add_argument('output',
                           help='.csv for representation or .npz for parsing')
    argparser.add_argument('-n','--Nmax',type=int,
                           help='maximum number of structures to be parsed')
    argparser.add_argument('--istart',type=int,
                           help='index to begin representation/parsing (\
                                 def 0)')
    argparser.add_argument('--iskip',type=int,
                           help='indices to skip between actions (def 0)')
    argparser.add_argument('-l','--loose',action="store_true",
                           help='parse loose .xyz files in input directory')
    argparser.add_argument('-p', '--propertyinput',
                           help='filename for property data to parse \
                                if separate from .xyz')
    argparser.add_argument('--ignore',action='append',
                           help='when parsing .xyz, ignore any lines \
                                 containing ignore string(s)')
    argparser.add_argument('--descriptor', choices=['BP'],
                           help='method of representing data')
    ## read in the arguments from the command line
    args = argparser.parse_args()

    represent = (args.action == 'represent')
    inputfile = args.input
    outputfile = args.output
    Nmax = args.Nmax
    istart = args.istart
    iskip = args.iskip
    loose = args.loose
    propertyfile = args.propertyinput
    ignore = args.ignore

    if represent:
        indata, outdata = readnpz(inputfile,Nmax,istart,iskip) #slice as needed
        species_all = get_species(indata)
        coords_all = get_coords(indata)
        species_set = sorted(set(np.concatenate(species_all)))  # alphabetical
        species_counts = np.asarray([[cstruct.count(cspecie)
                                      for cspecie in species_set]
                                    for cstruct in species_all])
        counts_max = np.amax(species_counts, axis=0)

        for species_count, elements, coords_list in zip(species_counts, species_all, coords_all):
            print('processing ' + str(j).rjust(10) + '/' + str(N), end='\r')
            sys.stdout.flush()
            try:
                data = pad(represent(coords_list, elements),
                           species_count, counts_max)

                #line = ','.join(data) + '\n'
                #fil.write(line)
                j += 1
            except:
                continue
        print(time.time() - t0)

        represent_data =
        descriptors_temp = [pad([represent(coords_list),
                                 species_count, counts_max)
                            for species_count, coords_list in
                            zip(species_counts, coords_all)]

    else:  #parse data
        if loose:
            separate_parsexyz(inputfile,Nmax,istart,iskip,ignore)
        else:
            together_parsexyz(inputfile,Nmax,istart,iskip,ignore)

        if propertyfile:
            parse_property(propertyfile,Nmax,istart,iskip)
        else:
            parse_property(inputfile,Nmax,istart,iskip)

        writestuff(outputfile)


    ## read in data from the xyz file(s) and store it in a numpy compressed zip file


    '''
    if convert:
        if propertyinput:
            properties = get_properties(propertyinput)
            structures = convert_xyz(inputfile)
        else:
            structures,energies = convert_xyz(inputfile, splitphrase)
        np.savez_compressed(arrayfile, structures=structures, energies=energies)
    else:
    ## extract arrays from the numpy compressed zip fileim[ort]
        arraydata = np.load(inputfile)
        structures = arraydata['structures']
        energies = arraydata['energies']

    N = min(len(structures),len(energies), int(Nmax))
    print(('Total structures: '+str(N)).ljust(50))
    
    ## convert structure coordinates into symmetry functions
    XYZ = parser(structures, N)
    

    t0 = time.time()
    j = 0
    with open(outputfile,'w') as fil:
        for (elements, coords) in XYZ:  # per structure
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
    '''
