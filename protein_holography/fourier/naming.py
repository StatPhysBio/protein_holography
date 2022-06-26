#
# naming.py -- Michael Pun -- January 2021
# Conventions and functions for naming the following types of data files:
#  - spherical projections of residue neighborhoods
#  - training logs from equivariant neural networks
#  - equivariant neural network weights
#

def get_id_from_params(args,ignore_params):
    params = {}
    args.k = [args.k[0],args.k[-1]]
    for arg in vars(args):
        if arg in ignore_params:
            continue
        if getattr(args,arg) == None:
            continue
        params[arg] = getattr(args,arg)
    
    tag = '_'.join(map(
            lambda x: str(x) + '=' + str(
                '+'.join(
                    map(str,params[x]))),
            sorted(params)))
    return tag.replace('(','').replace(')','')

def get_data_id(args):
    ignore_params = ['datadir','outputdir',
                     'verbosity','bsize',
                     'learnrate','hdim',
                     'nlayers','eVal',
                     'dataset','invariants',
                     'parallelism','easy',
                     'output','hdf5','input'] 
    return get_id_from_params(args,ignore_params)

def get_val_data_id(args):
    ignore_params = ['datadir','outputdir',
                     'verbosity','bsize',
                     'learnrate','hdim',
                     'nlayers','e',
                     'dataset','invariants',
                     'parallelism','easy','hdf5']  
    return get_id_from_params(args,ignore_params)

def get_network_id(args):
    ignore_params = ['datadir','outputdir','verbosity',
                     'dataset','invariants',
                     'parallelism','easy','hdf5',
                     'subset'
    ] 
    return get_id_from_params(args,ignore_params)

