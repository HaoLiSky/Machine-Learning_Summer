# Test command-line argument parser
argparser = initialize_argparser()
args = argparser.parse_args('parse au55.xyz energies.csv Ba,Ti,O -k energy'.split())

if args.output:
    out_name = args.output
    if '.hdf5' not in out_name:
            out_name = out_name + '.hdf5'
else:
    out_name = args.input.split('.')[0] + '.hdf5'
symbol_set = args.symbol_set.split(',')

print('action:',args.action,';','input:',args.input,';',
      'secondary input:',args.input2,';','system:',symbol_set,';',
      'output:',out_name,';','keyword:',args.keyword,';',
      'slices:',args.index,';','padding:',args.pad,';',
      'descriptor:',args.descriptor)

args = argparser.parse_args('represent au55.hdf5 BPparam.yaml Au,Pt -o fingerprints.hdf5 -i ::2 -a'.split())
if args.output:
    out_name = args.output
    if '.hdf5' not in out_name:
            out_name = out_name + '.hdf5'
else:
    out_name = args.input.split('.')[0] + '.hdf5'
symbol_set = args.symbol_set.split(',')

print('action:',args.action,';','input:',args.input,';',
      'secondary input:',args.input2,';','system:',symbol_set,';',
      'output:',out_name,';','keyword:',args.keyword,';',
      'slices:',args.index,';','padding:',args.pad,';',
      'descriptor:',args.descriptor)


# Test property file parsing
property_list = parse_property('OSZICAR', False, 'E0=', '::2')
print(len(property_list))
print(property_list[5:10])

# Test coordinate/species parsing & saving
parse_ase('XDATCAR', 'structures_test.hdf5', '::2', ['C','H','N','K','Bi','Cl'],
          property_list)

# Test file integrity
with h5py.File('structures_test.hdf5', 'r', libver='latest') as h5i:
    keys = list(h5i['structures'].keys())
    print('total structures:', len(keys))
    ind = np.random.randint(len(keys))
    print('randomly chosen index:', ind)
    s_name = 'XDATCAR_'+str(ind)
    dset = h5i['structures'][s_name]['coordinates']
    coords = dset[()]
    natoms = dset.attrs['natoms']
    symbol_set = [symbol.decode('utf-8')
                  for symbol in dset.attrs['symbol_set']]
    species_counts = dset.attrs['species_counts']
    propert = dset.attrs['property']
    print('coords shape:', coords.shape)
    print('atoms:', natoms)
    print('system:', symbol_set)
    print('species:', species_counts)
    print('energy:', propert)

# Test descriptor parsing & saving
apply_descriptors('structures_test.hdf5', '--', 'fingerprint_test.hdf5',
                  '::2', False, 'BP', ['C','H','N','K','Bi','Cl'])

# Test file integrity
with h5py.File('fingerprint_test.hdf5', 'r', libver='latest') as h5o:
    keys = list(h5o['structures'].keys())
    print('total structures:', len(keys))
    ind = np.random.randint(len(keys))
    print('randomly chosen index:', ind)
    s_name = 'XDATCAR_'+str(ind)
    dset1 = h5o['structures'][s_name]['pairs']
    dset2 = h5o['structures'][s_name]['triplets']
    g1 = dset1[()]
    g2 = dset2[()]
    natoms = dset1.attrs['natoms']
    symbol_set = [symbol.decode('utf-8')
                  for symbol in dset1.attrs['symbol_set']]
    species_counts = dset1.attrs['species_counts']
    propert = dset1.attrs['property']
    
    
    print('g1 shape:', g1.shape)
    print('g2 shape:', g2.shape)
    print('atoms:', natoms)
    print('system:', symbol_set)
    print('species:', species_counts)
    print('energy:', propert)
