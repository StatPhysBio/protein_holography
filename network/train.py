#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#

import tensorflow as tf
import numpy as np
import hnn
import os
import clebsch
from dataset import get_dataset, get_inputs
import sys, os
import logging
from argparse import ArgumentParser
import naming
import math
import tensorflow.keras.backend as K
import h5py
import sys
sys.path.append('/gscratch/spe/mpun/protein_holography/utils')
from posterity import get_metadata,record_metadata

logging.getLogger().setLevel(logging.INFO)
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=8000)])

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
                    help='L value for network')
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
parser.add_argument('--optimizer',
                    dest='opt',
                    type=str,
                    nargs='+',
                    default=['Adam'],
                    help='Optimizer to use')

args =  parser.parse_args()
    
# get metadata
#metadata = get_metadata()

# compile id tags for the data and the network to be trained
train_data_id = naming.get_data_id(args)
val_data_id = naming.get_val_data_id(args)
network_id = naming.get_network_id(args)
print(train_data_id)
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
print('Training dimensions:')
print(ds_train)
ds_train = ds_train.shuffle(1000000000)
ds_val = get_dataset(input_data_dir,val_data_id)
inputs,y_true = get_inputs(input_data_dir,train_data_id)

# get the number of classes directly from the dataset
for el in ds_val:
    n_classes = el[1].shape[0]
    break

# set up network
nlayers = args.nlayers[0]
hidden_l_dims = [[args.hdim[0]] * (args.netL[0] * 2)] * nlayers
print('hidden L dims',hidden_l_dims)
logging.info("L_MAX=%d, %d layers", args.netL[0], nlayers)
logging.info("Hidden dimensions: %s", hidden_l_dims) 
network = hnn.hnn(
    args.netL[0], hidden_l_dims, nlayers, n_classes,
    tf_cg_matrices, args.n_dense[0], 
    args.reg_strength[0], args.dropout_rate[0], args.scale[0])


@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)
@tf.function
def confidence(truth,pred):
    return tf.math.reduce_mean(tf.math.reduce_max(tf.nn.softmax(pred),axis=0))
    

optimizers = ['Adam','SGD']
if args.opt[0] not in optimizers:
    print(args.opt[0],' optimizer given not recognized')
if args.opt[0] == 'Adam':
    optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate[0])
if args.opt[0] == 'SGD':
    optimizer = tf.keras.optimizers.SGD(learning_rate=args.learnrate[0])


network.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics =['categorical_accuracy',
                          confidence])


# training dataset shouldn't be truncated unless testing
#ds_train_trunc = ds_train.batch(args.bsize[0]) #.take(50)
ds_val_trunc = ds_val.batch(128)

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
 
logs = args.outputdir + 'tb_test'

tboard_callback = tf.keras.callbacks.TensorBoard(log_dir = logs,
                                                 histogram_freq = 1,
                                                 profile_batch = '500,520')

# get initial batch size
bs = args.bsize[0]

# trainable weights
trainable_weights = network.trainable_weights
#print('Trainable weights')
#print(trainable_weights)
gradients = tf.reduce_mean(trainable_weights[0])
@tf.function
def norm_grad(truth,pred):
    # trainable weights 
    trainable_weights = network.trainable_weights

    with tf.GradientTape() as gt:
        gt.watch(inputs)
        outputs = network(inputs,training=False)
        loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = outputs)
    gradients = gt.gradient(loss,trainable_weights)

    total_grad = gradients[0]*gradients[0]
    total_grad = tf.reshape(total_grad,[tf.reduce_prod(total_grad.shape)])
    for i in range(1,len(gradients)):
        new_grad = gradients[i]*gradients[i]
        new_grad = tf.reshape(new_grad,[tf.reduce_prod(new_grad.shape)])
        total_grad = tf.concat([total_grad,new_grad],axis=0)
    total_grad = tf.math.reduce_variance(tf.math.sqrt(total_grad))
    
    #total_grad = tf.reduce_mean(tf.conc([(tf.math.sqrt(x*x)) for x in gradients]))

    return total_grad


network.compile(optimizer=optimizer,
                loss=loss_fn,
                metrics =['categorical_accuracy',
                          norm_grad,
                          confidence])
total_history = None
try:
    try:
        print('not loading')
#        network.load_weights(checkpoint_filepath)
    except Exception as e:
        print(e)
        logging.error("Unable to load weights.")

    while bs <= 0:
        # batch training set according to new batch size
        ds_train_trunc = ds_train.batch(bs)

        # stop after 3 epochs without decrease in loss
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=3, mode='min', min_delta=0.01)

        # # train network with new batch size
        try:
            history = network.fit(ds_train_trunc, epochs=100, shuffle=True,
                              validation_data=ds_val_trunc, 
                              verbose = args.verbosity,
                              callbacks=[model_checkpoint_callback,
                                         tboard_callback,
                                         early_stopping]
                              )
        except KeyboardInterrupt:
            print('Training interrupted...continuing')
        if total_history == None:
            total_history = history.history
        else:
            for k in total_history.keys():
                total_history[k] = total_history[k] + history.history[k]
        new_bs = bs*2
        print('Increasing batch size from {} to {}'.format(bs,new_bs))
        bs = new_bs



except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)
print('Total history:',total_history)

try:

    # batch training set according to new batch size
    ds_train_trunc = ds_train.batch(bs)
    
    # stop after 3 epochs without decrease in loss
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=200, mode='min', min_delta=0.0001)
    
    # train network with new batch size
    history = network.fit(ds_train_trunc, epochs=10000, shuffle=True,
                          validation_data=ds_val_trunc, 
                          verbose = args.verbosity,
                          callbacks=[model_checkpoint_callback,
                                     tboard_callback,
                                     early_stopping])
    print(history.history)
    for k in total_history.keys():
        total_history[k] = total_history[k] + history.history[k]



except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)

with h5py.File('../data/network/casp11_training30.hdf5','r+') as f:
    for k in total_history.keys():
        recorded = False
        run_num = 0
        while not recorded:
            try:
                dset = f.create_dataset(
                    '/{}/run_{}/{}'.format(network_id,run_num,k),
                    data=total_history[k]
                    )
                record_metadata(metadata,dset)
            except:
                run_num += 1
                continue
            recorded = True

# trainable weights
trainable_weights = network.trainable_weights

with tf.GradientTape() as gt:
    gt.watch(inputs)
    outputs = network(inputs,training=False)
    loss = tf.nn.softmax_cross_entropy_with_logits(labels = y_true,logits = outputs)
gradients = gt.gradient(loss,trainable_weights)

for w,g in zip(trainable_weights,gradients):
    print(w.name)
    g = g*g
    g = np.sqrt(g)
    print(tf.reduce_mean(g))

total_grad = np.mean(np.concatenate([(np.sqrt(x*x)).flatten() for x in gradients]))
print('Total grad = {}'.format(total_grad))

#print(gradients)
logging.info('Terminating successfully')

