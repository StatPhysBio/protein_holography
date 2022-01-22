#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#
from argparse import ArgumentParser
import logging
import math
import os
import sys

import keras.backend as K
import numpy as np
import tensorflow as tf

import protein_holography.network.clebsch as clebsch
from protein_holography.network.dataset import get_ss_dataset, get_ss_inputs
import protein_holography.network.hnn as hnn
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
parser.add_argument('--ss',
                    dest='ss',
                    type=bool,
                    nargs='+',
                    default=True,
                    help='ss tag')

args =  parser.parse_args()

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
ds_train = get_ss_dataset(input_data_dir, train_data_id)
ds_val = get_ss_dataset(input_data_dir,val_data_id)
inputs,y_true = get_ss_inputs(input_data_dir,train_data_id)

# get the number of classes directly from the dataset
for el in ds_train:
    n_classes = el[1].shape[0]
    print('number of classes = {}'.format(n_classes))
    break

# set up network
nlayers = args.nlayers[0]
hidden_l_dims = [[args.hdim[0]] * (args.netL[0] + 1)] * nlayers
logging.info("L_MAX=%d, %d layers", args.netL[0], nlayers)
logging.info("Hidden dimensions: %s", hidden_l_dims) 
print('nclasses finally = {}'.format(n_classes))
network = hnn.hnn(
    args.netL[0], hidden_l_dims, nlayers, n_classes,
    tf_cg_matrices, 1, args.scale[0])


@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)
@tf.function
def confidence(truth,pred):
    return tf.math.reduce_max(tf.nn.softmax(pred))
    

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate[0])
#optimizer = tf.keras.optimizers.SGD(learning_rate=args.learnrate[0])


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


# network.compile(optimizer=optimizer,
#                 loss=loss_fn,
#                 metrics =['categorical_accuracy',
#                           norm_grad,
#                           confidence])

try:
    try:
        print('not loading')
#        network.load_weights(checkpoint_filepath)
    except:
        logging.error("Unable to load weights.")
    # epoch_counter = 0
    # fails = 0
    # stepsize = 50
    # maxLR = 10*args.learnrate[0]
    # optLR = args.learnrate[0]
    while bs <= 3000:
        # batch training set according to new batch size
        ds_train_trunc = ds_train.batch(bs)

        # stop after 3 epochs without decrease in loss
        early_stopping = tf.keras.callbacks.EarlyStopping(
            monitor='loss', patience=3, mode='min', min_delta=0.0001)
        # cycle = np.floor(1 + epoch_counter / (2 * stepsize))
        # x = np.abs(epoch_counter/stepsize - 2 * cycle + 1)
        # curr_lr = optLR + (maxLR - optLR) * np.maximum(0, 1-x)
        # K.set_value(network.optimizer.learning_rate, curr_lr)
        # print('learnrate set to {}'.format(curr_lr))
        # # train network with new batch size
        history = network.fit(ds_train_trunc, epochs=10, shuffle=True,
                              validation_data=ds_val_trunc, 
                              verbose = args.verbosity,
                              callbacks=[model_checkpoint_callback,
                                         early_stopping])

        
#        print(history.history)
        # if len(history.history['loss']) < 50:
        #     fails += 1
        print('Increasing batch size from {} to {}'.format(bs,bs*8))
#        epoch_counter += 10
#        if fails > 5:
        bs = bs*8
        #fails = 0


except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)

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

