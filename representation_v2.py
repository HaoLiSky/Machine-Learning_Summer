import os, sys, argparse, time, re
import h5py
import numpy as np
from descriptors import BehlerParrinello
from scipy.spatial.distance import cdist
from itertools import combinations, combinations_with_replacement, islice
import ase

#BP_DEFAULT = [6.0, 1.0, 1.0, 1.0, 1.0]

def initialize_argparser():
    argparser = argparse.ArgumentParser(description='Converts structures to \
                                        symmetry functions.')
    argparser.add_argument('action', choices=['represent', 'parse'])
    argparser.add_argument('input',
                           help='.hdf5 for representation or ase-compatible \
                                 file or directory of files for parsing')
    argparser.add_argument('input2',
                           help='filename of property data to parse \
                           or parameters to use in fingerprint')
    argparser.add_argument('symbol_set', 
                           help='comma-separated list of unique elements; \
                           e.g. "Au" or "Ba,Ti,O"')
    argparser.add_argument('-k','--keyword',
                           help='keyword to parse embedded property data; \
                           default: None')
    argparser.add_argument('-o','--output',
                           help='specify filename for .hdf5 output; \
                                default mirrors input name')
    argparser.add_argument('-i', '--index', default=':',
                           help='slicing using numpy convention; \
                           e.g. ":250" or "-500:" or "::2"')
    argparser.add_argument('-a', '--pad', action='store_true',
                           help='pad output with None values')
    argparser.add_argument('-d','--descriptor', choices=['BP'], default='BP',
                           help='method of representing data; \
                           default: Parinnello-Behler')
    return argparser


def strslice(string):
    '''
    from ase.io.formats.py
    '''
    if ':' not in string:
        return int(string)
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return slice(*i)


def slice_generator(generator, string):
    '''
    from ase.io.formats.py
    '''
    if ':' not in string:
        return int(string)
    i = []
    for s in string.split(':'):
        if s == '':
            i.append(None)
        else:
            i.append(int(s))
    i += (3 - len(i)) * [None]
    return itertools.islice(generator, *i)


def parse_property(propname, loose, keyword, index):
    with open(propname,'r') as propfile:
        data = propfile.read()
    
    if loose:
        parser = re.compile('(\w+)(?:[=\s,:]+)(\S+)(?:[;,\s])')
        data_pairs = parser.findall(data)
        property_list = {k:v for k,v in data_pairs}
        #full dictionary, one value per structure file
    else:
        if keyword:
            parser = re.compile('(?:'+keyword
                                +'[=\s,:]*)(\S+)(?:[;,\s])')
        
            property_list = parser.findall(data)
            property_list = [float(match) for match in parser.findall(data)]
        else:
            parser = re.compile('(\S+)(?:[,;\s])')
            property_list = [float(match) for match in parser.findall(data)]
        # list of values after slicing
        
        # assumed to match with ase.io.iread's slicing
    return property_list[strslice(index)]


def write_structure(h5f, structure, symbol_set,symbol_order, s_name, 
                    property_value):
    coords = structure.get_positions(wrap=False)
    natoms = len(coords)
    species = structure.get_chemical_symbols()
    species_counts = np.asarray([species.count(cspecie)
                                 for cspecie
                                 in symbol_set]).astype('i4')
    species, coords = zip(*sorted(zip(species, coords),
                                  key=lambda x: symbol_order.get(x[0])))
    try:
        unit = structure.get_cell(complete=False)
        periodic = True
    except (ValueError, RuntimeError, AttributeError):
        unit = np.zeros((3,3))
        periodic = False
    dset_name = 'structures/%s/coordinates' % s_name
    dset_coords = h5f.create_dataset(dset_name,
                                     (natoms, 3),
                                     data=coords, dtype='f4',
                                     compression="gzip")
    dset_coords.attrs['natoms'] = natoms
    dset_coords.attrs['symbol_set'] = np.string_(symbol_set)
    dset_coords.attrs['species_counts'] = species_counts
    dset_coords.attrs['unit'] = unit
    dset_coords.attrs['periodic'] = periodic
    dset_coords.attrs['property'] = property_value


def parse_ase(inname, outname, index, symbol_set, property_list):
    prefix = inname+'_'

    symbol_order = {k:v for v,k in enumerate(symbol_set)}
    with h5py.File(outname, 'w', libver='latest') as h5f:
        nspecies = len(symbol_set)
        try:
            s_count = len(h5f['structures'].keys())
        except (KeyError):
            s_count = 0
        s_count_start = s_count+0
        print(str(s_count_start), 'structures to start')
        loose = os.path.isdir(inname)
        if loose:
            for root, dirs, files in slice_generator(os.walk(inname), index):
                for filename in files:
                    #print(os.path.join(root,filename))
                    try:
                        structures = ase.io.iread(filename)
                    except (ValueError, IOError, IndexError):
                        continue
                    for structure in structures:
                        s_name = filename+str(s_count)
                        property_value = property_list[filename]
                        write_structure(h5f, structure, symbol_set,
                                        symbol_order, s_name,
                                        property_value)
                        s_count += 1
        else:
            structures = ase.io.iread(inname, index=index)
            for structure in structures:
                s_name = prefix+str(s_count)
                property_value = property_list[s_count]
                write_structure(h5f, structure, symbol_set,
                                symbol_order, s_name, property_value)
                s_count += 1
        print(str(s_count-s_count_start), 'structures parsed into .hdf5')


def read_structure(h5i, s_name):
    dset = h5i['structures'][s_name]['coordinates']
    coords = dset[()]
    symbol_set = [symbol.decode('utf-8')
                  for symbol in dset.attrs['symbol_set']]
    species_counts = dset.attrs['species_counts']
    
    species_list = []
    for symbol, ccount in zip(symbol_set, species_counts):
        species_list += [symbol] * ccount
               
    unit = dset.attrs['unit']
    periodic = dset.attrs['periodic']
    property_value = dset.attrs['property']
    
    return [coords, symbol_set, species_counts, 
            species_list, unit, periodic, property_value]


def make_fingerprint(h5o, s_data, s_name, parameters):
    (coords, symbol_set, species_counts, 
     species_list, unit, periodic, property_value) = s_data
    
    '''
    bp = BehlerParrinello()
    
    ###### add updated with supercell code
    '''
    
    ### start placeholder code
    
    para_pairs, para_triplets = parameters
    n_atoms = len(coords)
    coord_sums = np.sum(coords, axis=1)
    # sum of coordinates for each atom
    n_species = len(symbol_set)
    
    pair_num = len(list(combinations_with_replacement(range(n_species), 1)))
    pair_factors = np.random.rand(pair_num, 1)
    # 1 factor per pair interaction (specie in sphere)
    para_factors_1 = np.sum(para_pairs, axis=0)
    para_num_1 = len(para_factors_1)
    # 1 factor per parameter set
    tile_atoms_1 = np.tile(coord_sums.reshape(n_atoms,1,1),
                           (1, pair_num, para_num_1))
    tile_inter_1 = np.tile(pair_factors.reshape(1,pair_num,1),
                           (n_atoms, 1, para_num_1))
    tile_parameters_1 = np.tile(para_factors_1.reshape(1,1, para_num_1),
                           (n_atoms, pair_num, 1))
    g_1 = np.multiply(np.multiply(tile_atoms_1, tile_inter_1), 
                      tile_parameters_1)
    #g_1.shape ~ (#atoms x #species x #pair_parameters)
                   
    triplet_num = len(list(combinations_with_replacement(range(n_species),
                                                         2)))
    triplet_factors = np.random.rand(triplet_num, 1)
    # 1 factor per triplet interaction (2 species in sphere)
    para_factors_2 = np.sum(para_triplets, axis=0)
    para_num_2 = len(para_factors_2)
    # 1 factor per parameter set
    tile_atoms_2 = np.tile(coord_sums.reshape(n_atoms,1,1),
                           (1, triplet_num,para_num_2))
    tile_inter_2 = np.tile(triplet_factors.reshape(1,triplet_num,1),
                           (n_atoms, 1, para_num_2))
    tile_parameters_2 = np.tile(para_factors_2.reshape(1,1, para_num_2),
                           (n_atoms, triplet_num, 1))
    g_2 = np.multiply(np.multiply(tile_atoms_2, tile_inter_2), 
                      tile_parameters_2)
    #g_2.shape ~ (#atoms x
    #             [#combinations of species with replacement] x
    #             #triplet_parameters
    
    ### end placeholder code
    
    dname_1 = 'structures/%s/pairs' % s_name
    dname_2 = 'structures/%s/triplets' % s_name
    dset_1 = h5o.create_dataset(dname_1,
                                g_1.shape,
                                data=g_1, dtype='f4',
                                compression="gzip")
    dset_2 = h5o.create_dataset(dname_2,
                                g_2.shape,
                                data=g_2, dtype='f4',
                                compression="gzip")
    dset_1.attrs['natoms'] = len(coords)
    dset_1.attrs['symbol_set'] = np.string_(symbol_set)
    dset_1.attrs['species_counts'] = species_counts
    dset_1.attrs['property'] = property_value


def apply_descriptors(inname, BPparafile, outname, index, pad, 
                      descriptor, symbol_set):
    
    #parameters = read_BP_parameters(BPparafile)
    kp = 4
    kt = 6
    parameters = (np.random.rand(5,kp), np.random.rand(5,kt))
    #temp!
    
    with h5py.File(inname, 'r', libver='latest') as h5i:
        with h5py.File(outname, 'w', libver='latest') as h5o:
            s_names = list(h5i['structures'].keys())
            s_tot = len(s_names)
            s_count = 0
            f_count = 0
            for j, s_name in enumerate(s_names):
                print('processing',str(j+1).rjust(10),'/',
                      str(s_tot).rjust(10),end='\r')
                try:
                    s_data = read_structure(h5i, s_name)
                    make_fingerprint(h5o, s_data, s_name, parameters)
                    s_count += 1
                except (KeyError, ValueError):
                    f_count += 1
                    continue
            print(str(s_count), 'fingerprints created')
            if f_count > 0:
                print(str(f_count), 'fingerprint(s) failed')


if __name__ == '__main__':
    argparser = initialize_argparser()
    args = argparser.parse_args()
    ## read in the arguments from the command line
    
    symbol_set = args.symbol_set.split(',')
    if not len(symbol_set) == len(set(list(symbol_set))):
        raise ValueError('duplicate elements in symbol_set: '+symbol_set)
        
    if ':' not in index:
        raise ValueError('invalid slice specified: '+args.index)

    in_name = args.input
    if args.output:
        out_name = args.output
        if '.hdf5' not in out_name:
            out_name = out_name + '.hdf5'
    else:
        out_name = in_name.split('.')[0] + '.hdf5'

    if args.action == 'represent':
        if '.hdf5' not in in_name:
            raise IOError('Expected .hdf5 input')
        
        t0 = time.time()
        apply_descriptors(in_name, args.input2, out_name, args.index, 
                          args.pad, args.descriptor, symbol_set)
        print(time.time() - t0)
        
    elif args.action == 'parse':
        loose = os.path.isdir(inname)
        property_list = parse_property(args.input2, loose, args.keyword
                                       index=args.index)
        #dictionary if loose, list otherwise
        parse_ase(in_name, out_name, args.index, symbol_set, property_list)


