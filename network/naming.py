#
# naming.py -- Michael Pun -- January 2021
# Conventions and functions for naming the following types of data files:
#  - spherical projections of residue neighborhoods
#  - training logs from equivariant neural networks
#  - equivariant neural network weights
#

def get_id_from_params(args,ignore_params):
    params = {}
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
                     'scale','load'] 
    return get_id_from_params(args,ignore_params)

def get_val_data_id(args):
    args.e = args.eVal
    ignore_params = ['datadir','outputdir',
                     'verbosity','bsize',
                     'learnrate','hdim',
                     'nlayers','eVal',
                     'scale','load'] 
    return get_id_from_params(args,ignore_params)

def get_network_id(args):
    ignore_params = ['datadir','outputdir','verbosity','learnrate','bsize','load']
    return get_id_from_params(args,ignore_params)

