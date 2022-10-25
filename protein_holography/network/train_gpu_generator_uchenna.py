#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with given parameters,
# loads holograms from .npy files, and then tests the network via a function call.
#
from argparse import ArgumentParser
import atexit
import math
import logging
import os
import subprocess
import sys

import h5py
import numpy as np
import tensorflow as tf
import tensorflow.keras.backend as K
from tqdm import tqdm

import protein_holography.network.clebsch as clebsch
from protein_holography.network.dataset import get_dataset, get_inputs
from protein_holography.network.hnn import hnn
from protein_holography.network.naming import get_network_id
#sys.path.append('/premiumproteindatadrive/protein_holography/utils')
from protein_holography.utils.posterity import get_metadata, record_metadata

def main():
    print('Arg 0 is : ',sys.argv[0],'\n\n')
    singular_pdbs = [b'3A0M', b'1AW8', b'3JQH', b'3NIR', b'3O2R', b'3O2R', b'1GTV', b'2W8S', b'3JQH', b'3NIR', b'4AON', b'3L4J', b'2HAL', b'2ZS0', b'2W8S', b'3L4J', b'4BY8', b'4AON', b'2HAL', b'1GTV', b'2ZS0', b'1AW8', b'3A0M', b'4BY8',b'2J6V']

    aa_to_one_letter = {'ALA': 'A', 'CYS': 'C', 'ASP': 'D', 'GLU': 'E',
                           'PHE': 'F', 'GLY': 'G', 'HIS': 'H', 'ILE': 'I',
                            'LYS': 'K', 'LEU': 'L', 'MET': 'M', 'ASN': 'N',
                            'PRO': 'P', 'GLN': 'Q', 'ARG': 'R', 'SER':'S',
                            'THR': 'T', 'VAL': 'V', 'TRP': 'W', 'TYR': 'Y'}
    one_letter_to_aa = {'A': 'ALA',
     'C': 'CYS',
     'D': 'ASP',
     'E': 'GLU',
     'F': 'PHE',
     'G': 'GLY',
     'H': 'HIS',
     'I': 'ILE',
     'K': 'LYS',
     'L': 'LEU',
     'M': 'MET',
     'N': 'ASN',
     'P': 'PRO',
     'Q': 'GLN',
     'R': 'ARG',
     'S': 'SER',
     'T': 'THR',
     'V': 'VAL',
     'W': 'TRP',
     'Y': 'TYR'}
    aa_to_ind_size = {'CYS': 2, 'ILE': 8, 'GLN': 12, 'VAL': 6, 'LYS': 13,
           'PRO': 4, 'GLY': 0, 'THR': 5, 'PHE': 16, 'GLU': 14,
           'HIS': 15, 'MET': 11, 'ASP': 7, 'LEU': 9, 'ARG': 17,
           'TRP': 19, 'ALA': 1, 'ASN': 10, 'TYR': 18, 'SER': 3}
    ind_to_aa_size = {0: 'GLY', 1: 'ALA', 2: 'CYS', 3: 'SER', 4: 'PRO',
           5: 'THR', 6: 'VAL', 7: 'ASP', 8: 'ILE', 9: 'LEU',
           10: 'ASN', 11: 'MET', 12: 'GLN', 13: 'LYS', 14: 'GLU',
           15: 'HIS', 16: 'PHE', 17: 'ARG', 18: 'TYR', 19: 'TRP'}
    aa_to_ind_one_letter = {'ALA': 0, 'CYS': 1, 'ASP': 2, 'GLU': 3,
                            'PHE': 4, 'GLY': 5, 'HIS': 6, 'ILE': 7,
                            'LYS': 8, 'LEU': 9, 'MET': 10, 'ASN': 11,
                            'PRO': 12, 'GLN': 13, 'ARG': 14, 'SER':15,
                            'THR': 16, 'VAL': 17, 'TRP': 18, 'TYR': 19}
    background_freqs = {'ALA': 7.4, 'CYS': 3.3, 'ASP': 5.9, 'GLU': 3.7,
                            'PHE': 4., 'GLY': 7.4, 'HIS': 2.9, 'ILE': 3.8,
                            'LYS': 7.2, 'LEU': 7.6, 'MET': 1.8, 'ASN': 4.4,
                            'PRO': 5., 'GLN': 5.8, 'ARG': 4.2, 'SER': 8.1,
                            'THR': 6.2, 'VAL': 6.8, 'TRP': 1.3, 'TYR': 3.3}
    ind_to_aa_one_letter = {0: 'ALA', 1: 'CYS', 2: 'ASP', 3: 'GLU',
                            4: 'PHE', 5: 'GLY', 6: 'HIS', 7: 'ILE',
                            8: 'LYS', 9: 'LEU', 10: 'MET', 11: 'ASN',
                            12: 'PRO', 13: 'GLN', 14: 'ARG', 15: 'SER', 
                            16: 'THR', 17: 'VAL', 18: 'TRP', 19: 'TYR'}
    aa_to_ind_hydro = {'ALA': 8, 'ARG': 15, 'ASN': 17, 'ASP': 14,
                       'CYS': 6, 'GLN': 13, 'GLU': 10, 'GLY': 11,
                       'HIS': 18, 'ILE': 1, 'LEU': 0, 'LYS': 16,
                       'MET': 5, 'PHE': 2, 'PRO': 19, 'SER': 12,
                       'THR': 9, 'TRP': 3, 'TYR': 7, 'VAL': 4}
    ind_to_aa_hydro = {8: 'ALA', 15: 'ARG', 17: 'ASN', 14: 'ASP',
                       6: 'CYS', 13: 'GLN', 10: 'GLU', 11: 'GLY',
                       18: 'HIS', 1: 'ILE', 0: 'LEU', 16: 'LYS', 
                       5: 'MET', 2: 'PHE', 19: 'PRO', 12: 'SER',
                       9: 'THR', 3: 'TRP', 7: 'TYR', 4: 'VAL'}

    logging.getLogger().setLevel(logging.INFO)
    gpus = tf.config.list_physical_devices('GPU')
    #if gpus:
    #    for gpu in gpus:
    #        tf.config.experimental.set_virtual_device_configuration(gpu,[tf.config.experimental.VirtualDeviceConfiguration(memory_limit=2000)])

    # parameters 
    parser = ArgumentParser()
    parser.add_argument('--outputdir',
                        dest='outputdir',
                        type=str,
                        default='../output',
                        help='data directory')
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
    parser.add_argument('--connection',
                        dest='connection',
                        type=str,
                        nargs='+',
                        default=None,
                        help='type of connection for the nonlinearity')
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
    parser.add_argument('--datadir',
                        dest='datadir',
                        type=str,
                        default='../data',
                        help='data directory')
    parser.add_argument('--loaddir',
                        dest='loaddir',
                        type=str,
                        default='',
                        help='data from which to load weights')
    
    args =  parser.parse_args()

    # get metadata
    #metadata = get_metadata()

    # compile id tags for the data and the network to be trained
    network_id = get_network_id(args)

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
    print('Training dimensions:')


    n_classes = 20



    class DataGenerator(tf.keras.utils.Sequence):
        'Generates data for Keras'
        def __init__(self, list_IDs, labels, hdf5_file, hdf5_dataset, batch_size=32, 
                     n_classes=20, shuffle=True):
            'Initialization'
            self.batch_size = batch_size
            self.labels = labels
            self.list_IDs = list_IDs
            print('Number of data points = ',len(list_IDs))
            self.hdf5_file = hdf5_file
            self.hdf5_dataset = hdf5_dataset
            with h5py.File(self.hdf5_file,'r') as f:
                print(f[self.hdf5_dataset])
                self.dt = f[self.hdf5_dataset][0].dtype
                self.entire_f = f[self.hdf5_dataset][:]
            self.n_classes = n_classes
            self.shuffle = shuffle
            self.on_epoch_end()
            self.L_max = args.netL[0]
            self.noise_vals = np.load('/gscratch/scrubbed/mpun/data/casp12/zernikegrams/mean_power.npy',allow_pickle=True)[()]
            
        def __len__(self):
            'Denotes the number of batches per epoch'
            return int(np.floor(len(self.list_IDs) / self.batch_size))

        def __getitem__(self, index):
            'Generate one batch of data'
            # Generate indexes of the batch
            indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

            # Find list of IDs
            list_IDs_temp = [self.list_IDs[k] for k in indexes]

            # Generate data
            X, y = self.__data_generation(list_IDs_temp)

            return X, y

        def on_epoch_end(self):
            'Updates indexes after each epoch'
            self.indexes = np.arange(len(self.list_IDs))
            if self.shuffle == True:
                np.random.shuffle(self.indexes)

        def __data_generation(self, list_IDs_temp):
            'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
            # Initialization
            X = np.empty((self.batch_size),dtype=self.dt)
            y = np.empty((self.batch_size), dtype=int)
            X_noise = [
                np.random.normal(scale=1e-1,size=(self.batch_size,147,2*l+1)) +
                1j * np.random.normal(scale=1e-1,size=(self.batch_size,147,2*l+1))
                for l in range(self.L_max + 1)
            ]
            
            # Generate data
            #with h5py.File(self.hdf5_file,'r') as f:
            hfile = h5py.File(self.hdf5_file, 'r')
            for i, ID in enumerate(list_IDs_temp):
                # Store sample
                    #print(ID)
    #                 try:
                X[i,] = self.entire_f[ID]
                #X[i,] = hfile[self.hdf5_dataset][ID]
    #                 except:
    #                     print(ID)
                    # Store class
                y[i] = self.labels[ID]
            for i in range(self.L_max + 1):
                X[str(i)] += X_noise[i] * self.noise_vals[i][None,:,None]
            #print(X.dtype)
            #print("---X Y SIZES HERE----")
            #print((X.size * X.itemsize)/1024**2)
            #print((y.size * y.itemsize)/1024**2)
            return {i:X[str(i)].astype(np.complex64) for i in range(self.L_max + 1)}, tf.keras.utils.to_categorical(y, num_classes=self.n_classes)

    #
    # Get test holograms
    #


    #
    # validation holograms
    #
    max_val = int(3e6)
    filename = "/gscratch/scrubbed/mpun/data/casp12/zernikegrams/casp12_chain_validation_zernikegrams_round2_no_RBD_T4.hdf5"
    #filename = './data/validation_zernikegrams.hdf5'
    with h5py.File(filename ,'r') as f:
        #print(ids.shape)
        ids = np.array(f['nh_list'][:])
        total_ids = f['nh_list'].shape[0]
        bad_idxs = np.hstack(
            (np.squeeze(np.argwhere(ids[:,0]==b'Z')),
             np.squeeze(np.argwhere(ids[:,0]==b'X')),
             np.squeeze(np.argwhere(ids[:,0]==b''))
                             )
        )
        good_idxs = np.setdiff1d(np.arange(total_ids,dtype=int),bad_idxs)

    np.random.seed(0)
    np.random.shuffle(good_idxs)
    subset = good_idxs[:max_val]


    print(len(subset))
    #[idxs.remove(x) for x in tqdm(bad_idxs)]
    labels = {idx:aa_to_ind_size[one_letter_to_aa[x.decode('utf-8')]] 
              for idx,x in tqdm(zip(subset,ids[subset,0]))}
    val_dg = DataGenerator(subset,labels,filename,
                           'validation_chains',
                           batch_size=512)





    # set up network
    nlayers = args.nlayers[0]
    hidden_l_dims = [[args.hdim[0]] * (args.netL[0] * 2)] * nlayers
    print('hidden L dims',hidden_l_dims)
    logging.info("L_MAX=%d, %d layers", args.netL[0], nlayers)
    logging.info("Hidden dimensions: %s", hidden_l_dims) 
    network = hnn(
        args.netL[0], hidden_l_dims, nlayers, n_classes,
        tf_cg_matrices, args.n_dense[0], 
        args.reg_strength[0], args.dropout_rate[0], args.scale[0], args.connection[0])


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

    batch = val_dg.__getitem__(0)
    network.evaluate(x=batch[0],y=batch[1],use_multiprocessing=False,workers=1)
    network.summary()


    logging.info('Training network')

    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)

    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, mode='min', min_delta=0.01)


    # get initial batch size
    bs = args.bsize[0]


    network.compile(optimizer=optimizer,
                    loss=loss_fn,
                    metrics =['categorical_accuracy',
                              #norm_grad,
                              confidence])
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.1,
                                  patience=10, min_lr=1e-9)

    # loading of weights
    if args.loaddir != '':
        load_file = [x for x in os.listdir(args.loaddir) if 'index' in x and network_id.split('learnrate')[0] in x][0]
        load_file = load_file.strip('.index')
        print(load_file)
        load_checkpoint_filepath = os.path.join(
            args.loaddir,
            load_file
        )
        try:
            network.load_weights(load_checkpoint_filepath)
            print('Successfully loaded weights')
        except:
            print('Unable to load weights')
    else:
        print('Not attempting to load weights')
    total_history = None
    #max_train = int(6e6)
    # filename = "./data/training_zernikegrams.hdf5"
    filename = "/gscratch/scrubbed/mpun/data/casp12/zernikegrams/casp12_chain_training_zernikegrams_round2_no_RBD_T4.hdf5"
    with h5py.File(filename ,'r') as f:
        ids = np.array(f['nh_list'][:])
        total_ids = f['nh_list'].shape[0]
        bad_idxs = np.hstack(
            (np.squeeze(np.argwhere(ids[:,0]==b'Z')),
             np.squeeze(np.argwhere(ids[:,0]==b'X')),
             np.squeeze(np.argwhere(ids[:,0]==b''))
                             )
        )
        bad_idxs = np.hstack(
            (bad_idxs,
             np.hstack([np.squeeze(np.argwhere(ids[:,1]==pdb)) for pdb in np.unique(singular_pdbs)])
            )
        )
        good_idxs = np.setdiff1d(np.arange(total_ids,dtype=int),bad_idxs)

    np.random.shuffle(good_idxs)
    subset = good_idxs#[:max_train]
    print(len(subset))
    #subsets = np.split(subset[:int(6e6)],100)
    #subsets.append(subset[-93:])

    labels = {idx:aa_to_ind_size[one_letter_to_aa[x.decode('utf-8')]] 
                 for idx,x in tqdm(zip(subset,ids[subset,0]))}
    dg = DataGenerator(subset,labels,filename,
                       'training_chains',
                       #'pdb_subsets/img=x-ray diffraction_max_res=2.5/split_0.8_0.2_0.0/'
                       #'train/pdbs',
                       batch_size=args.bsize[0])

    history = network.fit(
        dg,
        use_multiprocessing=True,
        workers=40,
        validation_data=val_dg,
        validation_steps=100,
        epochs=300,
        verbose = args.verbosity,
        max_queue_size=900,
        steps_per_epoch=1000,
        callbacks=[
            model_checkpoint_callback,
            reduce_lr,
            #tboard_callback,
            early_stopping
        ]
    )
    print('Done training')
    np.save(args.outputdir + '/' + network_id + '_history.npy',
            history.history,
            allow_pickle=True)
    print('Done saving')
    
    print('Killing subprocesses to free up GPU and end python subprocesses')
    

    python_procs = subprocess.Popen( "ps aux | grep python",
                                     stdin=subprocess.PIPE, shell=True, stdout=subprocess.PIPE )
    output = python_procs.stdout.read()
    lookatme = [o.split() for o in output.split(b'\n')]
    for proc in lookatme[::-1]:
        if len(proc) < 13:
            continue
        if proc[12] == sys.argv[0].encode():
            os.kill(int(proc[1].decode('utf-8')), 9)
    print('Done killing subprocesses')
    
if __name__ == "__main__":
    main()
    sys.exit()
    




