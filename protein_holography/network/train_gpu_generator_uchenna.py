#
# File to train networks built from the hnn.py class. 
#
# This file establishes the clebsch gordan coefficients, sets up an hnn with
# given parameters, loads holograms from .npy files, and then tests the network 
# via a function call.
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
#from tqdm import tqdm

import protein_holography.network.clebsch as clebsch
from protein_holography.network.dataset import get_dataset, get_inputs
from protein_holography.network.hnn import hnn
from protein_holography.network.naming import get_network_id
from protein_holography.utils.posterity import get_metadata, record_metadata
from protein_holography.utils.protein import one_letter_to_aa, aa_to_ind_size
from protein_holography.utils.zernikegram import (
    get_channel_dict,channel_freq_to_inds)
singular_pdbs = [
        b'3A0M', b'1AW8', b'3JQH', b'3NIR', b'3O2R', b'3O2R', b'1GTV',
        b'2W8S', b'3JQH', b'3NIR', b'4AON', b'3L4J', b'2HAL', b'2ZS0',
        b'2W8S', b'3L4J', b'4BY8', b'4AON', b'2HAL', b'1GTV', b'2ZS0',
        b'1AW8', b'3A0M', b'4BY8', b'2J6V'
]

def restrict_channels(
    structured_array: np.ndarray,
    channels: np.ndarray,
    L_max: int
) -> np.ndarray:
    """
    Restrict an array to a set of channels
    
    Parameters
    ----------
    structred_array : np.ndarray
        A structured array storing a zernikegram
    channels : np.ndarray
        An int array of the indices to keep
    L_max : int
        The highest spherical order used in the zernikegram
    """
    ds_size = structured_array.shape[0
    ]
    num_channels = len(channels)
    dt = np.dtype(
        [(str(l),'complex64',(num_channels,2*l+1))
            for l in range(0,L_max + 1)])
    new_arr = np.zeros(shape=(ds_size),dtype=dt)
    for l in range(0,L_max + 1):
        new_arr[str(l)][:] = structured_array[str(l)][:,channels]
    return new_arr

class DataGenerator(tf.keras.utils.Sequence):
    """
    DataGenerator class for loading data at train time

    Attributes
    ----------
    batch_size : int
    labels : list
    list_IDs : list
    hdf5_file : str
    hdf5_dataset : str
    dt : 
    entire_f : np.ndarray
    n_classes : int
    shuffle : bool
    L_max : int
    noise_vals : n.ndarray

    Methods
    -------
    """
    def __init__(
        self, list_IDs, labels, hdf5_file, hdf5_dataset, netL,
        batch_size=32, n_classes=20, channels=None, shuffle=True
    ):
        'Initialization'
        self.batch_size = batch_size
        self.labels = labels
        self.list_IDs = list_IDs
        logging.info(
            f"DataGenerator with number of data points = {len(list_IDs)}")
        self.hdf5_file = hdf5_file
        self.hdf5_dataset = hdf5_dataset
        self.L_max = netL
        with h5py.File(self.hdf5_file,'r') as f:
            self.entire_f = f[self.hdf5_dataset][:]
            if channels == None:
                self.dt = f[self.hdf5_dataset].dtype
            else:
                self.entire_f = restrict_channels(
                    self.entire_f,
                    channels,self.L_max
                )
                self.dt = self.entire_f.dtype
        self.n_classes = n_classes
        self.shuffle = shuffle
        self.on_epoch_end()
        # self.noise_vals = np.load(
        #     '/gscratch/scrubbed/mpun/data/casp12/zernikegrams/mean_power.npy',
        #     allow_pickle=True)[()]
        
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))


    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[
            index*self.batch_size:(index+1)*self.batch_size]

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
        bs = np.min([self.batch_size,len(list_IDs_temp)])

        # Initialization
        X = np.zeros((bs),dtype=self.dt)
        y = np.zeros((bs), dtype=int)

        ## Declaring noise
        # X_noise = [
        #     np.random.normal(scale=1e-1,size=(bs,147,2*l+1)) +
        #     1j * np.random.normal(scale=1e-1,size=(bs,147,2*l+1))
        #     for l in range(self.L_max + 1)
        # ]
        
        # Generate data
        for i, ID in enumerate(list_IDs_temp):
            X[i,] = self.entire_f[ID]
            y[i] = self.labels[ID]

        ## Adding noise
        # for i in range(self.L_max + 1):
        #     X[str(i)] += X_noise[i] * self.noise_vals[i][None,:,None]
        
        return (
            {i:X[str(i)].astype(np.complex64) for i in range(self.L_max + 1)},
            tf.keras.utils.to_categorical(y, num_classes=self.n_classes))

def set_GPU_usage(mem_limit):
        
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
       for gpu in gpus:
           tf.config.experimental.set_virtual_device_configuration(
               gpu,
               [tf.config.experimental.VirtualDeviceConfiguration(
                   memory_limit=mem_limit)]
            )
    logging.info(
        "GPUs Available: %d", 
        len(tf.config.experimental.list_physical_devices('GPU'))
    )
    #tf.config.threading.set_intra_op_parallelism_threads(4)
    #tf.config.threading.set_inter_op_parallelism_threads(4)

def kill_subprocesses():
    """Kill subprocesses to free up GPU if running sequential jobs"""
    logging.info('Killing subprocesses to free up GPU and end python subprocesses')
    

    python_procs = subprocess.Popen(
        "ps aux | grep python", stdin=subprocess.PIPE, shell=True, 
        stdout=subprocess.PIPE )
    output = python_procs.stdout.read()
    lookatme = [o.split() for o in output.split(b'\n')]
    for proc in lookatme[::-1]:
        if len(proc) < 13:
            continue
        if proc[12] == sys.argv[0].encode():
            os.kill(int(proc[1].decode('utf-8')), 9)
    logging.info('Done killing subprocesses')

def get_clean_dataset(
    filename: str,
    dataset: str,
    netL: int,
    bsize: int=256,
    channels=None,
    shuffle: bool=True,
):
    with h5py.File(filename ,'r') as f:
        #print(ids.shape)
        ids = np.array(f['nh_list'][:])
        total_ids = f['nh_list'].shape[0]
        bad_idxs = np.hstack(
            (np.squeeze(np.argwhere(ids[:,0]==b'Z')),
             np.squeeze(np.argwhere(ids[:,0]==b'X')),
             np.squeeze(np.argwhere(ids[:,0]==b'')))
        )
        bad_idxs = np.hstack(
            (bad_idxs,
             np.hstack(
                [np.squeeze(np.argwhere(ids[:,1]==pdb)) 
                for pdb in np.unique(singular_pdbs)])
            )
        )
        good_idxs = np.setdiff1d(np.arange(total_ids,dtype=int),bad_idxs)

    if shuffle:
        np.random.seed(0)
        np.random.shuffle(good_idxs)
    subset = good_idxs[:]

    # dict[idx:label]
    labels = {idx:aa_to_ind_size[one_letter_to_aa[x.decode('utf-8')]] 
              for idx,x in zip(subset,ids[subset,0])}

    return DataGenerator(
        subset, labels, filename, dataset, netL, 
        batch_size=bsize, channels=channels)

def train_network(
    network_id: str,
    datadir: str,
    l: int,
    outputdir: str,
    channels: str,
    nlayers: int,
    hdim: int,
    netL: int,
    n_dense: int,
    reg_strength: float,
    dropout_rate: float,
    scale: float,
    connection: str,
    opt: str,
    learnrate: float,
    bsize: int,
    loaddir: str,
    verbosity: int,
):
    """
    Train a network

    Arguments
    ---------

    Returns
    -------

    """
    # load clebsch-gordan coefficients
    cg_file = os.path.join("../utils/CG_matrix_l=13.npy")
    tf_cg_matrices = clebsch.load_clebsch(cg_file, l)

    # declare filepaths for loading inputs and saving network weights
    #input_data_dir = os.path.join(args.datadir, args.proj[0])
    checkpoint_filepath = os.path.join(outputdir, network_id)

    n_classes = 20
    channel_inds = None
    if channels !=  None:
        channel_inds = channel_freq_to_inds(channels)

    logging.info("Collecting validation dataset")
    val_filename = "/gscratch/scrubbed/mpun/data/casp12/zernikegrams/"\
               "casp12_chain_validation_zernikegrams_round2_no_RBD_T4.hdf5"
    val_dataset = 'validation_chains'
    # val_filename = (
    #     "/gscratch/scrubbed/mpun/data/RBD_Ab/zernikegrams/"
    #     "RBD_Ab_zernikegrams.hdf5"
    # )
    # val_dataset = 'RBD'
    val_dg = get_clean_dataset(
        val_filename,
        val_dataset,
        netL,
        bsize=1024,
        channels=channel_inds,
        shuffle=True
    )

    logging.info("Collecting training dataset")
    train_filename = (
        "/gscratch/scrubbed/mpun/data/casp12/zernikegrams/"
        "casp12_chain_training_zernikegrams_round2_no_RBD_T4.hdf5"
    )
    train_dataset = 'training_chains'
    # train_filename = (
    #     "/gscratch/scrubbed/mpun/data/RBD_Ab/zernikegrams/"
    #     "RBD_Ab_zernikegrams.hdf5"
    # )
    # train_dataset = 'RBD'
    train_dg = get_clean_dataset(
        train_filename,
        train_dataset,
        netL,
        bsize=256,
        channels=channel_inds,
        shuffle=True
    )

    logging.info("Establishing network")
    hidden_l_dims = [[hdim] * (netL * 2)] * nlayers
    logging.info("L_MAX=%d, %d layers", netL, nlayers)
    logging.info("Hidden dimensions: %s", hidden_l_dims) 
    network = hnn(
        netL, hidden_l_dims, nlayers, n_classes,
        tf_cg_matrices, n_dense, 
        reg_strength, dropout_rate, scale, connection)


    @tf.function
    def loss_fn(truth, pred):
        return tf.nn.softmax_cross_entropy_with_logits(
            labels = truth,
            logits = pred)
    @tf.function
    def confidence(truth,pred):
        return tf.math.reduce_mean(
            tf.math.reduce_max(tf.nn.softmax(pred),axis=0))


    optimizers = ['Adam','SGD']
    if opt not in optimizers:
        logging.error(opt,' optimizer given not recognized')
    if opt == 'Adam':
        optimizer = tf.keras.optimizers.Adam(learning_rate=learnrate)
    if opt == 'SGD':
        optimizer = tf.keras.optimizers.SGD(learning_rate=learnrate)


    network.compile(
        optimizer=optimizer, loss=loss_fn,
        metrics =['categorical_accuracy',confidence])

    # Evaluate network to establish graph and print summary
    batch = val_dg.__getitem__(0)
    network.evaluate(x=batch[0],y=batch[1],use_multiprocessing=False,workers=1)
    network.summary()


    
    # callbacks for monitoring
    model_checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_filepath,
        save_weights_only=True,
        monitor='loss',
        mode='min',
        save_best_only=True)
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss', patience=20, mode='min', min_delta=0.01)
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss', factor=0.1, patience=10, min_lr=1e-9)
    # network.compile(optimizer=optimizer,
    #                 loss=loss_fn,
    #                 metrics =['categorical_accuracy',
    #                           confidence])


    # loading of weights
    if loaddir != '':
        logging.info(
            "Attempting to load weights from previously trained network")
        load_file = [
            x for x in os.listdir(loaddir) 
            if 'index' in x and network_id.split('learnrate')[0] in x][0]
        load_file = load_file.strip('.index')
        print(load_file)
        load_checkpoint_filepath = os.path.join(
            loaddir,
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

    
    logging.info('Training network')
    history = network.fit(
        train_dg,
        use_multiprocessing=True,
        workers=40,
        validation_data=val_dg,
        validation_steps=100,
        epochs=300,
        verbose = verbosity,
        max_queue_size=900,
        steps_per_epoch=1000,
        callbacks=[
            model_checkpoint_callback,
            reduce_lr,
            #tboard_callback,
            early_stopping
        ]
    )

    logging.info('Done training network')
    np.save(outputdir + '/' + network_id + '_history.npy',
            history.history,
            allow_pickle=True)
    logging.info('Done saving network history')
    
    ## No need to kill subprocesses on hyak
    ## Useful if ever running on Azure or node without slurm
    #kill_subprocesses()


def main():
    print('Arg 0 is : ',sys.argv[0],'\n\n')
    
    logging.getLogger().setLevel(logging.INFO)

    # parameters 
    parser = ArgumentParser()
    parser.add_argument(
        '--outputdir', dest='outputdir', type=str, default='../output',
        help='data directory')
    parser.add_argument(
        '--projection', dest='proj', type=str, nargs='+', default=None,
        help='Radial projection used (e.g. hgram, zgram)')
    parser.add_argument(
        '--netL', dest='netL', type=int, nargs='+', default=None,
        help='L value for network')
    parser.add_argument(
        '-l', dest='l', type=int, nargs='+', default=None,
        help='L value for practical cutoff (L <= file_L')
    parser.add_argument(
        '-k', dest='k', type=complex, nargs='+', default=None,
        help='k value')
    parser.add_argument(
        '-d', dest='d', type=float, nargs='+', default=None,
        help='d value')
    parser.add_argument(
        '--rmax', dest='rmax', type=float, default=None, nargs='+',
        help='rmax value')
    parser.add_argument(
        '--ch', dest='ch', type=str, nargs='+', default=None,
        help='ch value')
    parser.add_argument(
        '--verbosity', dest='verbosity', type=int, default=1,
        help='Verbosity mode')
    parser.add_argument(
        '--hdim', dest='hdim', type=int, nargs='+', default=None,
        help='hidden dimension size')
    parser.add_argument(
        '--nlayers', dest='nlayers', type=int, nargs='+', default=None,
          help='num layers')
    parser.add_argument(
        '--bsize', dest='bsize', type=int, nargs='+', default=None,
         help='training minibatch size')
    parser.add_argument(
        '--learnrate', dest='learnrate', type=float, nargs='+', default=None,
         help='learning rate')
    parser.add_argument(
        '--aas', dest='aas', type=str, nargs='+', default=None,
         help='aas for fewer class classifier')
    parser.add_argument(
        '--scale', dest='scale', type=float, nargs='+', default=None,
         help='scale for rescaling inputs')
    parser.add_argument(
        '--connection', dest='connection', type=str, nargs='+', default=None,
         help='type of connection for the nonlinearity')
    parser.add_argument(
        '--dropout', dest='dropout_rate', type=float, nargs='+', default=None,
         help='rate for dropout')
    parser.add_argument(
        '--reg', dest='reg_strength', type=float, nargs='+', default=None,
         help='strength for regularization (typically l1 or l2')
    parser.add_argument(
        '--n_dense', dest='n_dense', type=int, nargs='+', default=None,
        help='number of dense layers to put at end of network')
    parser.add_argument(
        '--optimizer', dest='opt', type=str, nargs='+', default=['Adam'],
         help='Optimizer to use')
    parser.add_argument(
        '--datadir',  dest='datadir', type=str, default='../data', 
        help='data directory')
    parser.add_argument(
        '--loaddir', dest='loaddir', type=str, default='',
         help='data from which to load weights')
    parser.add_argument(
        '--channels', dest='channels', type=str, default=None, nargs='+',
        help='data from which to load weights')
    args =  parser.parse_args()

    # get metadata
    metadata = get_metadata()

    # compile id tags for the data and the network to be trained
    if args.channels != None:
        network_id = get_network_id(args)
        print(args.k)
        channels_dict = get_channel_dict(args.channels[0],int(np.max(args.k)))
    else: 
        channel_dict = None

    print('channels dict is',channels_dict)
    train_network(
        network_id,
        args.datadir,
        args.l[0],
        args.outputdir,
        channels_dict,
        args.nlayers[0],
        args.hdim[0],
        args.netL[0],
        args.n_dense[0],
        args.reg_strength[0],
        args.dropout_rate[0],
        args.scale[0],
        args.connection[0],
        args.opt[0],
        args.learnrate[0],
        args.bsize[0],
        args.loaddir,
        args.verbosity
    )
if __name__ == "__main__":
    main()
    sys.exit()
    




