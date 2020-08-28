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
from dataset import get_dataset
import sys, os
import logging

logging.getLogger().setLevel(logging.INFO)

logging.info("GPUs Available: %d", len(tf.config.experimental.list_physical_devices('GPU')))
tf.config.threading.set_intra_op_parallelism_threads(4)
tf.config.threading.set_inter_op_parallelism_threads(1)

L_MAX = 6
# load premade holograms
k = 0.001
d = 10.0
examples_per_aa = 1000
examples_per_aa_val = 1000
d_val = 10.0

cg_file = '../CG_matrix_l=10.npy'
hologram_dir = "../holograms"
checkpoint_filepath = './saved_weights/weights'

tf_cg_matrices = clebsch.load_clebsch(cg_file, L_MAX)

# network parameters
num_layers = 4
num_aa = 20
hidden_l_dims = [[10] * (L_MAX + 1)] * num_layers
logging.info("L_MAX=%d, %d layers", L_MAX, num_layers)
logging.info("Hidden dimensions: %s", hidden_l_dims)
network = hnn.hnn(L_MAX, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)

network.compile(optimizer=optimizer, loss=loss_fn, metrics =['categorical_accuracy'])

ds_train = get_dataset(hologram_dir, examples_per_aa, k, d, L_MAX)
ds_val = get_dataset(hologram_dir, examples_per_aa_val, k, d, L_MAX)

# training dataset shouldn't be truncated unless testing
ds_train_trunc = ds_train.batch(2).take(50)
ds_val_trunc = ds_val.batch(2).take(10)

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
    monitor='val_loss', patience=1, mode='min', min_delta=0.0001)

try:
    try:
        network.load_weights(checkpoint_filepath)
    except:
        logging.error("Unable to load weights.")
    network.fit(ds_train_trunc, epochs=10, shuffle=True,
                validation_data=ds_val_trunc, 
                callbacks=[model_checkpoint_callback, early_stopping])
except KeyboardInterrupt:
    logging.warning("KeyboardInterrupt received. Exiting.")
    sys.exit(os.EX_SOFTWARE)

logging.info('Terminating successfully')
