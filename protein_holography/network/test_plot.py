
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

import matplotlib as mpl
from matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import protein_holography.network.clebsch as clebsch
import protein_holography.network.hnn as hnn
import protein_holography.utils.protein as protein
from protein_holography.network.dataset import get_dataset, get_hgram_labels

logging.getLogger().setLevel(logging.INFO)
#default_datadir = os.path.join(os.path.dirname(__file__), "../data/")
#default_outputdir = os.path.join(os.path.dirname(__file__), "../output/")

# for testing
default_datadir = '/gscratch/spe/mpun/protein_holography/data'
default_outputdir = '/gscratch/spe/mpun/protein_holography/output'


parser = ArgumentParser()
parser.add_argument('-L',
                    dest='L',
                    type=int,
                    default=6,
                    help='L value')
parser.add_argument('-k',
                    dest='k',
                    type=float,
                    default=1.,
                    help='k value')
parser.add_argument('--ks',
                    dest='ks',
                    nargs='+',
                    type=float,
                    default=1.,
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
                    default='elnc',
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
                    default=4,
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


args =  parser.parse_args()

logging.info("GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
#tf.config.threading.set_intra_op_parallelism_threads(4)
#tf.config.threading.set_inter_op_parallelism_threads(4)

cg_file = os.path.join(args.datadir, "CG_matrix_l=13.npy")
hologram_dir = os.path.join(args.datadir, "holograms")

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

tf_cg_matrices = clebsch.load_clebsch(cg_file, args.L)

# network parameters
num_layers = args.nlayers
num_aa = 20
hidden_l_dims = [[args.hdim] * (args.L + 1)] * num_layers
logging.info("L_MAX=%d, %d layers", args.L, num_layers)
logging.info("Hidden dimensions: %s", hidden_l_dims)
network = hnn.hnn(args.L, hidden_l_dims, num_layers, num_aa, tf_cg_matrices,1)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=args.learnrate)


network.compile(optimizer=optimizer, loss=loss_fn, metrics =['categorical_accuracy'])

test_hgrams,test_labels = get_hgrams_labels(hologram_dir, args.ch, args.e_val, args.L, args.ks, args.d, args.rH)

print(test_hgrams)

logging.info('Testing network')


network.load_weights(checkpoint_filepath)
total = len(test_hgrams[0])
print('Total number of examples = ' + str(total))
prediction = network.predict(test_hgrams,batch_size=len(test_hgrams[0]))
print('predicted')
print(prediction)
c_mat = np.zeros([20,20])

print(prediction[0])
print(np.argmax(prediction[0]))
print(test_labels[0])
correct = 0.
for i in range(total):
    y_true = np.argmax(test_labels[i])
    y_pred = np.argmax(prediction[i])
    if y_true == y_pred:
        correct += 1.
    c_mat[y_true][y_pred] += 1.
print('Accuracy: ' + str(correct/total))


test_tag = "ch={}_e={}_e_val={}_l={}_k={}_d={}_rH={}_nlayers={}_hdim={}".format(args.ch,
                                                              args.e, args.e_val,
                                                              args.L, args.k,
                                                              args.d, args.rH,
                                                              args.nlayers,args.hdim)

c_mat = c_mat/total*20.
np.save('/usr/lusers/mpun/plots/c_mat_val_' + test_tag, c_mat,allow_pickle=True)
size = 20
from matplotlib.pyplot import figure
figure(num=None, figsize=(8, 6), dpi=1000, facecolor='w', edgecolor='k')
plt.imshow(np.power(c_mat,0.5), cmap='hot', interpolation='nearest',vmin=0.,vmax=1.)
plt.xticks(range(size),[ind_to_aa[i] for i in range(size)],rotation=90,fontsize=10)
plt.yticks(range(size),[ind_to_aa[i] for i in range(size)],fontsize=10)
plt.xlabel('Predicted amino acid',fontsize=12)
plt.ylabel('Input amino acid',fontsize=12)
plt.title('Square root training prediction matrix',fontsize=15)
cbar = plt.colorbar()
cbar.ax.set_ylabel(r'$\sqrt{accuracy}$', rotation=270,fontsize=12,labelpad=20)
plt.savefig('/usr/lusers/mpun/plots/val_' + test_tag + '.svg',dpi=1000, format='svg')

logging.info('Terminating successfully')

