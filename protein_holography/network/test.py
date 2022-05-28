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
from protein_holography.network.dataset import get_dataset
import protein_holography.network.hnn as hnn

logging.getLogger().setLevel(logging.INFO)
default_datadir = os.path.join(os.path.dirname(__file__), "../data/")
default_outputdir = os.path.join(os.path.dirname(__file__), "../output/")

parser = ArgumentParser()
parser.add_argument('-L',
                    dest='L',
                    type=int,
                    default=6,
                    help='L value')
parser.add_argument('-k',
                    dest='k',
                    type=float,
                    default=0.001,
                    help='k value')
parser.add_argument('-d',
                    dest='d',
                    type=float,
                    default=10.0,
                    help='d value')
parser.add_argument('-e',
                    dest='e',
                    type=int,
                    default=1000,
                    help='examples per aminoacid')
parser.add_argument('--e_val',
                    dest='e_val',
                    type=int,
                    default=1000,
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


args =  parser.parse_args()

logging.info("GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.threading.set_inter_op_parallelism_threads(4)

cg_file = os.path.join(args.datadir, "CG_matrix_l=13.npy")
hologram_dir = os.path.join(args.datadir, "holograms")

checkpoint_filepath = os.path.join(
    args.outputdir,
    "e={}_k={}_d={}_l={}/weights".format(args.e, args.k, args.d, args.L))

tf_cg_matrices = clebsch.load_clebsch(cg_file, args.L)

# network parameters
num_layers = 4
num_aa = 20
hidden_l_dims = [[10] * (args.L + 1)] * num_layers
logging.info("L_MAX=%d, %d layers", args.L, num_layers)
logging.info("Hidden dimensions: %s", hidden_l_dims)
network = hnn.hnn(args.L, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

network.compile(optimizer=optimizer, loss=loss_fn, metrics =['categorical_accuracy'])

ds_train = get_dataset(hologram_dir, args.e, args.k, args.d, args.L)
ds_val = get_dataset(hologram_dir, args.e_val, args.k, args.d, args.L)

# training dataset shouldn't be truncated unless testing
ds_train_trunc = ds_train.batch(4) #.take(50)
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
    monitor='loss', patience=3, mode='min', min_delta=0.0001)

try:
    try:
        network.load_weights(checkpoint_filepath)
    except:
        logging.error("Unable to load weights.")
    network.evaluate(ds_val_trunc,verbose = args.verbosity)
    network.evaluate(ds_train_trunc, verbose = args.verbosity)

except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)

logging.info('Terminating successfully')
