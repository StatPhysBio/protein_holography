#
# naming.py -- Michael Pun -- January 2021
# Conventions and functions for naming the following types of data files:
#  - spherical projections of residue neighborhoods
#  - training logs from equivariant neural networks
#  - equivariant neural network weights
#
import copy

def get_id_from_params(args,ignore_params,network=False):
    params = {}
    #if network:
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
                     'nlayers','eVal','eTest',
                     'scale','load',
                     'netL','dropout_rate',
                     'n_dense','reg_strength',
                     'opt'
] 
    return get_id_from_params(args,ignore_params)

def get_val_data_id(args):
    val_args = copy.deepcopy(args)
    val_args.e = args.eVal
    ignore_params = ['datadir','outputdir',
                     'verbosity','bsize',
                     'learnrate','hdim',
                     'nlayers','eVal','eTest',
                     'scale','load',
                     'netL','dropout_rate',
                     'netL','dropout_rate',
                     'n_dense','reg_strength',
                     'opt'
] 
    return get_id_from_params(val_args,ignore_params)

def get_test_data_id(args):
    val_args = copy.deepcopy(args)
    val_args.e = args.eTest
    ignore_params = ['datadir','outputdir',
                     'verbosity','bsize',
                     'learnrate','hdim',
                     'nlayers','eVal','eTest',
                     'scale','load',
                     'netL','dropout_rate',
                     'netL','dropout_rate',
                     'n_dense','reg_strength',
                     'opt'
] 
    return get_id_from_params(val_args,ignore_params)

def get_network_id(args):
    ignore_params = ['datadir','outputdir','verbosity','load','eTest']
    return get_id_from_params(args,ignore_params,network=True)


def get_test_network_id(args):
    ignore_params = ['datadir','outputdir','verbosity','load']
    return get_id_from_params(args,ignore_params,network=True)

