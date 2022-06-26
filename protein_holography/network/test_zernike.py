#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#

from argparse import ArgumentParser
import logging
import os
import sys

import numpy as np
import tensorflow as tf

import protein_holography.network.clebsch as clebsch
import protein_holography.network.hnn as hnn
from protein_holography.network.dataset import get_dataset, get_dataset_zernike

logging.getLogger().setLevel(logging.INFO)
#default_datadir = os.path.join(os.path.dirname(__file__), "../data/")
#default_outputdir = os.path.join(os.path.dirname(__file__), "../output/")

# for testing
default_datadir = '/gscratch/spe/mpun/protein_holography/data'
default_outputdir = '/gscratch/spe/mpun/protein_holography/output'


parser = ArgumentParser()
parser.add_argument('--file_L',
                    dest='file_L',
                    type=int,
                    default=6,
                    help='L value for file specification')
parser.add_argument('-L',
                    dest='L',
                    type=int,
                    default=6,
                    help='L value for practical cutoff (L <= file_L')
parser.add_argument('-k',
                    dest='k',
                    type=int,
                    default=1,
                    help='k value')
parser.add_argument('--ks',
                    dest='ks',
                    nargs='+',
                    type=float,
                    default=[],
                    help='multiple k values')
parser.add_argument('-d',
                    dest='d',
                    type=float,
                    default=10.0,
                    help='d value')
parser.add_argument('--rH',
                    dest='rH',
                    type=float,
                    default=5.0,
                    help='rH value')
parser.add_argument('--ch',
                    dest='ch',
                    type=str,
                    default='el',
                    help='ch value')
parser.add_argument('-e',
                    dest='e',
                    type=int,
                    default=1024,
                    help='examples per aminoacid')
parser.add_argument('--e_val',
                    dest='e_val',
                    type=int,
                    default=128,
                    help='examples per aminoacid validation')
parser.add_argument('--datadir',
                    dest='datadir',
                    type=str,
                    default=default_datadir,
                    help='data directory')
parser.add_argument('--outputdir',
                    dest='outputdir',
                    type=str,
                    default=default_outputdir,
                    help='data directory')
parser.add_argument('--verbosity',
                    dest='verbosity',
                    type=int,
                    default=1,
                    help='Verbosity mode')
parser.add_argument('--hdim',
                    dest='hdim',
                    type=int,
                    default=10,
                    help='hidden dimension size')
parser.add_argument('--nlayers',
                    dest='nlayers',
                    type=int,
                    default=3,
                    help='num layers')
parser.add_argument('--bsize',
                    dest='bsize',
                    type=int,
                    default=16,
                    help='training minibatch size')
parser.add_argument('--learnrate',
                    dest='learnrate',
                    type=float,
                    default=0.001,
                    help='learning rate')
parser.add_argument('--aas',
                    dest='aas',
                    type=str,
                    nargs='+',
                    default=[],
                    help='aas for fewer class classifier')

args =  parser.parse_args()
print(args.ks)
logging.info("GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.threading.set_inter_op_parallelism_threads(4)

cg_file = os.path.join(args.datadir, "CG_matrix_l=13.npy")
hologram_dir = os.path.join(args.datadir, "zernikegrams")

checkpoint_filepath = os.path.join(
    args.outputdir,
    "ch={}_e={}_e_val={}_l={}_k={}_d={}_rH={}_nlayers={}_hdim={}/weights".format(args.ch,
                                                              args.e, args.e_val,
                                                              args.L, args.k,
                                                              args.d, args.rH,
                                                              args.nlayers,args.hdim))
if args.k == -1:
    print('here')
    checkpoint_filepath = os.path.join(
        args.outputdir,
        "ch={}_e={}_e_val={}_l={}_k={}_d={}_rH={}_nlayers={}_hdim={}/weights".format(args.ch,
                                                                                     args.e, args.e_val,
                                                                                     args.L, args.ks,
                                                                                     args.d, args.rH,
                                                                                     args.nlayers,args.hdim))
if len(args.aas) > 0:
    print('aas')
    checkpoint_filepath = os.path.join(
        args.outputdir,
        "ch={}_e={}_e_val={}_l={}_k={}_d={}_rH={}_nlayers={}_hdim={}_aas={}/weights".format(args.ch,
                                                                                     args.e, args.e_val,
                                                                                     args.L, args.k,
                                                                                     args.d, args.rH,
                                                                                               args.nlayers,args.hdim,
                                                                                               args.aas))



tf_cg_matrices = clebsch.load_clebsch(cg_file, args.L)

# network parameters
num_layers = args.nlayers
num_aa = 20
if len(args.aas) > 0:
    num_aa = len(args.aas)
hidden_l_dims = [[args.hdim] * (args.L + 1)] * num_layers
logging.info("L_MAX=%d, %d layers", args.L, num_layers)
logging.info("Hidden dimensions: %s", hidden_l_dims)
network = hnn.hnn(args.L, hidden_l_dims, num_layers, num_aa, tf_cg_matrices,1,True)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)
@tf.function
def confidence(y_true, y_pred):
#    return tf.einsum('ij,ij->',y_true,tf.nn.softmax(y_pred))
    return tf.math.reduce_max(tf.nn.softmax(y_pred))

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate)

network.compile(optimizer=optimizer, loss=loss_fn, metrics =['categorical_accuracy',tf.keras.losses.CategoricalCrossentropy(from_logits=True),confidence])

if args.k != -1:
    ds_train = get_dataset_zernike(hologram_dir, args.ch, args.e, args.file_L, args.k, args.d, args.rH, args.aas)
    ds_val = get_dataset_zernike(hologram_dir, args.ch, args.e_val, args.file_L, args.k, args.d, args.rH, args.aas)

if args.k == -1:
    ds_train = get_dataset_zernike(hologram_dir, args.ch, args.e, args.file_L, args.ks, args.d, args.rH, args.aas)
    ds_val = get_dataset_zernike(hologram_dir, args.ch, args.e_val, args.file_L, args.ks, args.d, args.rH, args.aas)

# training dataset shouldn't be truncated unless testing
ds_train_trunc = ds_train.batch(args.bsize)#.take(50)
ds_val_trunc = ds_val.batch(2)

network.evaluate(ds_train.batch(1).take(1))
network.summary()

logging.info('Training network')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=20, mode='min', min_delta=0.0001)


print(checkpoint_filepath)
network.load_weights(checkpoint_filepath)



print(network.evaluate(ds_train.batch(1).take(1),return_dict=True))
    #for x in ds_train.batch(1):
    #    print(network.evaluate(x=x[0],y=x[1],return_dict=True))
    # train network
#except KeyboardInterrupt:
#    logging.warning("KeyboardInterrupt received. Exiting.")
#    sys.exit(os.EX_SOFTWARE)

logging.info('Terminating successfully')
