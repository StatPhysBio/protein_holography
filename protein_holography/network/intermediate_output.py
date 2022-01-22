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
import protein_holography.network.hnn_intermediate as hnn_inter
import protein_holography.network.naming as naming

logging.getLogger().setLevel(logging.INFO)

# parameters 
parser = ArgumentParser()
parser.add_argument('--projection',
                    dest='proj',
                    type=str,
                    nargs='+',
                    default=None,
                    help='Radial projection used (e.g. hgram, zgram)')
parser.add_argument('--netL',
                    dest='netL',
                    type=int,
                    nargs='+',
                    default=None,
                    help='L value for L max in network')
parser.add_argument('-l',
                    dest='l',
                    type=int,
                    nargs='+',
                    default=None,
                    help='L value for practical cutoff (L <= file_L')
parser.add_argument('-k',
                    dest='k',
                    type=complex,
                    nargs='+',
                    default=None,
                    help='k value')
parser.add_argument('-d',
                    dest='d',
                    type=float,
                    nargs='+',
                    default=None,
                    help='d value')
parser.add_argument('--rH',
                    dest='rH',
                    type=float,
                    default=None,
                    nargs='+',
                    help='rH value')
parser.add_argument('--rmax',
                    dest='rmax',
                    type=float,
                    default=None,
                    nargs='+',
                    help='rmax value')
parser.add_argument('--ch',
                    dest='ch',
                    type=str,
                    nargs='+',
                    default=None,
                    help='ch value')
parser.add_argument('-e',
                    dest='e',
                    type=int,
                    nargs='+',
                    default=None,
                    help='examples per aminoacid')
parser.add_argument('--eVal',
                    dest='eVal',
                    type=int,
                    nargs='+',
                    default=None,
                    help='examples per aminoacid validation')
parser.add_argument('--datadir',
                    dest='datadir',
                    type=str,
                    default='../data',
                    help='data directory')
parser.add_argument('--outputdir',
                    dest='outputdir',
                    type=str,
                    default='../output',
                    help='data directory')
parser.add_argument('--verbosity',
                    dest='verbosity',
                    type=int,
                    default=1,
                    help='Verbosity mode')
parser.add_argument('--hdim',
                    dest='hdim',
                    type=int,
                    nargs='+',
                    default=None,
                    help='hidden dimension size')
parser.add_argument('--nlayers',
                    dest='nlayers',
                    type=int,
                    nargs='+',
                    default=None,
                    help='num layers')
parser.add_argument('--bsize',
                    dest='bsize',
                    type=int,
                    nargs='+',
                    default=None,
                    help='training minibatch size')
parser.add_argument('--learnrate',
                    dest='learnrate',
                    type=float,
                    nargs='+',
                    default=None,
                    help='learning rate')
parser.add_argument('--aas',
                    dest='aas',
                    type=str,
                    nargs='+',
                    default=None,
                    help='aas for fewer class classifier')
parser.add_argument('--scale',
                    dest='scale',
                    type=float,
                    nargs='+',
                    default=None,
                    help='scale for rescaling inputs')
parser.add_argument('--load',
                    dest='load',
                    type=bool,
                    default=False,
                    help='choose to load weights')
parser.add_argument('--dropout',
                    dest='dropout_rate',
                    type=float,
                    nargs='+',
                    default=None,
                    help='rate for dropout')
parser.add_argument('--reg',
                    dest='reg_strength',
                    type=float,
                    nargs='+',
                    default=None,
                    help='strength for regularization (typically l1 or l2')
parser.add_argument('--n_dense',
                    dest='n_dense',
                    type=int,
                    nargs='+',
                    default=None,
                    help='number of dense layers to put at end of network')

args =  parser.parse_args()

# compile id tags for the data and the network to be trained
train_data_id = naming.get_data_id(args)
val_data_id = naming.get_val_data_id(args)
network_id = naming.get_network_id(args)
print(network_id)
logging.info("GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.threading.set_inter_op_parallelism_threads(4)

# load clebsch-gordan coefficients
cg_file = os.path.join(args.datadir, "CG_matrix_l=13.npy")
tf_cg_matrices = clebsch.load_clebsch(cg_file, args.l[0])

# declare filepaths for loading inputs and saving network weights
input_data_dir = os.path.join(args.datadir, args.proj[0])
checkpoint_filepath = os.path.join(
    args.outputdir,
    network_id)


# parameters for the network
ds_train = get_dataset(input_data_dir, train_data_id)
ds_val = get_dataset(input_data_dir,val_data_id)

# get the number of classes directly from the dataset
for el in ds_val:
    n_classes = el[1].shape[0]
    break

# set up network
nlayers = args.nlayers[0]
hidden_l_dims = [[args.hdim[0]] * (args.netL[0] + 1)] * nlayers
logging.info("L_MAX=%d, %d layers", args.netL[0], nlayers)
logging.info("Hidden dimensions: %s", hidden_l_dims) 
network = hnn.hnn(
    args.netL[0], hidden_l_dims, nlayers, n_classes,
    tf_cg_matrices, args.n_dense[0], args.reg_strength[0], args.dropout_rate[0],
    args.scale[0])
#intermediate_layer = 1
intermediate_network = hnn_inter.hnn_intermediate(
    args.netL[0], hidden_l_dims, nlayers,
    n_classes, tf_cg_matrices, args.n_dense[0],
    args.reg_strength[0], args.dropout_rate[0], args.scale[0])
#    intermediate_layer)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)
@tf.function
def confidence(truth,pred):
    return tf.math.reduce_max(tf.nn.softmax(pred))

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate[0])

network.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics =['categorical_accuracy',confidence])

@tf.function
def inter_loss_fn(truth, pred):
    return 1.


intermediate_network.compile(optimizer=optimizer,
                loss=inter_loss_fn)


# training dataset shouldn't be truncated unless testing
ds_train_trunc = ds_train.batch(args.bsize[0]) #.take(50)
ds_val_trunc = ds_val.batch(2)
#intermediate_network.evaluate(ds_train.batch(1).take(1))
network.evaluate(ds_train.batch(1).take(1))
network.summary()

# attempt to get intermediate layers from the network
# for x in network.layers[:-1]:
#     print(x)
#     print(x.__dict__.keys())
#     print('Output = ' + str(x.output_))
#     print('\n\n\n')

#print(network.model())
logging.info('Getting prediction from network')

model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=checkpoint_filepath,
    save_weights_only=True,
    monitor='loss',
    mode='min',
    save_best_only=True)

early_stopping = tf.keras.callbacks.EarlyStopping(
    monitor='loss', patience=20, mode='min', min_delta=0.0001)

try:
    try:
        if args.load:
            network.load_weights(checkpoint_filepath)
            intermediate_network.load_weights(checkpoint_filepath)
        else:
            print('not loading')
    except:
        logging.error("Unable to load weights.")
    network.save_weights('temp_weights')
    intermediate_network.load_weights('temp_weights')
    prediction = network.predict(ds_val.batch(1),
                          callbacks=[model_checkpoint_callback,
                                     early_stopping])
    intermediate_out = intermediate_network.predict(ds_val.batch(1))
    print(prediction[0])
#    print(intermediate_out)
    if args.load:
        np.save(args.outputdir + '/loaded_predictions_' + network_id,
                prediction)
        np.save(args.outputdir + '/loaded_intermediates_' + network_id,
                intermediate_out)
    else:
        np.save(args.outputdir + '/predictions_' + network_id,
                prediction)
        np.save(args.outputdir + '/intermediates_' + network_id,
                intermediate_out)

except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)
network.evaluate(ds_val.batch(128))

logging.info('Terminating successfully')

