import tensorflow as tf
import numpy as np
import linearity
import nonlinearity
import hnn
import os

cg_file = 'CG_matrix_l=5.npy'
cg_matrices = np.load(cg_file, allow_pickle=True).item()
L_MAX = 3
tf_cg_matrices = {}
for l in range(L_MAX + 1):
    for l1 in range(L_MAX + 1):
        for l2 in range(0,l1+1):
            tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                    

print('Training hnn model')
num_layers = 3
L_MAX = 3
hidden_l_dims = []
for i in range(num_layers):
    curr_l_dims = []
    for l in range(L_MAX + 1):
        curr_l_dims.append(np.random.randint(low=1,high=5))
    hidden_l_dims.append(curr_l_dims)
print('Making network with L_MAX=' + str(L_MAX) + ' with '  + str(num_layers) + ' layers with hidden dimensions ' + 
      str(hidden_l_dims))
num_aa = 20
network = hnn.hnn(L_MAX, hidden_l_dims, num_layers, num_aa, tf_cg_matrices)

print('Loading test input to model')
batch_size = 7

train_hgrams_real = np.load('train_hgram_real_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams_imag = np.load('train_hgram_imag_example_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy'
                            ,allow_pickle=True,encoding='latin1')[()]
train_hgrams = {}
for l in range(L_MAX + 1):
    train_hgrams[l] = tf.convert_to_tensor(train_hgrams_real[l] + 1j * train_hgrams_imag[l],
                                           dtype=tf.dtypes.complex64)
labels = tf.convert_to_tensor(np.load('train_labels_examplesPerAA=20_k=0.0001_d=0.0_l=3.npy',
                 allow_pickle=True,encoding='latin1'))
print(labels)
@tf.function
def loss_fn(truth, pred):
    return tf.nn.softmax_cross_entropy_with_logits(
        labels = truth,
        logits = pred)

optimizer = tf.keras.optimizers.Adam(learning_rate=1.e-3)
#network.compile(optimizer=optimizer, loss=loss_fn)

print('The shape is')
print(train_hgrams)
#print(train_hgrams)

print('Running network via predict')
print(network.predict(train_hgrams))

print('Training network')
network.fit(x=train_hgrams,y=labels)


print('Terminating successfully')
