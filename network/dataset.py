import numpy as np
import tensorflow as tf
import numpy as np
import scipy
import naming
import wigner

def get_dataset(data_dir, data_id):

    hgrams = np.load('/'.join([data_dir,data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')[()]
    print('/'.join([data_dir,data_id + '.npy']))
#     # assemble complex holograms from real and imaginary parts
#     hgrams = {}
#     for l in range(l_max + 1):
#         hgrams[l] = (hgrams_real[l] + 
#                      1j * hgrams_imag[l]).astype("complex64")
    
    labels = np.load('/'.join([data_dir,'labels_' + data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((hgrams,labels))

def get_inputs(data_dir, data_id):

    hgrams = np.load('/'.join([data_dir,data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')[()]
    for i in hgrams.keys():
        hgrams[i] = tf.convert_to_tensor(hgrams[i])
    print('/'.join([data_dir,data_id + '.npy']))

    labels = np.load('/'.join([data_dir,'labels_' + data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')

    return hgrams,labels

def get_equivariance_test_dataset(data_dir, data_id, L_MAX):
    alpha = np.random.uniform(0,2*np.pi)
    beta = np.random.uniform(0,np.pi)
    gamma = np.random.uniform(0,2*np.pi)
    print('Rotation: ' +str((alpha,beta,gamma)))
    hgrams = np.load('/'.join([data_dir,data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')[()]
    print('/'.join([data_dir,data_id + '.npy']))
    eq_hgrams = {}
    for i in range(L_MAX+1):
        eq_hgrams[i] = []
        eq_hgrams[i].append(hgrams[i][0].astype('complex64'))
#        eq_hgrams[i].append(hgrams[i][0].astype('complex64'))
        eq_hgrams[i].append(
            np.einsum(   
                'mn,cm->cn',
                wigner.wigner_d_matrix(i,alpha,beta,gamma),
                hgrams[i][0]
                )
            )
        
        eq_hgrams[i] = np.array(eq_hgrams[i],dtype='complex64')
#    print('Eq hgrams = ' + str(eq_hgrams[0]))
    labels = np.load('/'.join([data_dir,'labels_' + data_id + '.npy']),
                     allow_pickle=True,
                     encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((eq_hgrams,labels[0:2]))


if __name__ == "__main__":
    ds = get_dataset('../holograms', 1000, 0.0001, 10.0, 6)
    print(ds.batch(1).batch(2).take(2))

def get_dataset_delta(hologram_dir, ch, examples_per_aa, l_max, k, d, rH, aas):
    p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_rH={}.npy".format(
        hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH)
    if len(aas) > 0:
        p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_rH={}_aas={}.npy".format(
            hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH, aas)


    hgrams_real = np.load(p('dgram_real'), allow_pickle=True,encoding='latin1')[()]
    hgrams_imag = np.load(p('dgram_imag'), allow_pickle=True,encoding='latin1')[()]
    hgrams = {}
    for l in range(l_max + 1):
        hgrams[l] = (hgrams_real[l] + 1j * hgrams_imag[l]).astype("complex64")

    #print(hgrams[1][0])
    labels = np.load(p('dgram_labels'), allow_pickle=True,encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((hgrams,labels))

def get_dataset_zernike(hologram_dir, ch, examples_per_aa, l_max, k, d, rH, aas):
    p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}.npy".format(
        hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH)
    if len(aas) > 0:
        p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_aas={}.npy".format(
            hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH, aas)


    hgrams_real = np.load(p('zgram_real'), allow_pickle=True,encoding='latin1')[()]
    hgrams_imag = np.load(p('zgram_imag'), allow_pickle=True,encoding='latin1')[()]
    hgrams = {}
    for l in range(l_max + 1):
        hgrams[l] = (hgrams_real[l] + 1j * hgrams_imag[l]).astype("complex64")

    #print(hgrams[1][0])
    labels = np.load(p('zgram_labels'), allow_pickle=True,encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((hgrams,labels))

if __name__ == "__main__":
    ds = get_dataset('../holograms', 1000, 0.0001, 10.0, 6)


def get_hgrams_labels(hologram_dir, ch, examples_per_aa, l_max, k, d, rH):
    p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_rH={}.npy".format(
        hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH)

    hgrams_real = np.load(p('hgram_real'), allow_pickle=True,encoding='latin1')[()]
    hgrams_imag = np.load(p('hgram_imag'), allow_pickle=True,encoding='latin1')[()]
    hgrams = {}
    for l in range(l_max + 1):
        hgrams[l] = (hgrams_real[l] + 1j * hgrams_imag[l]).astype("complex64")

    labels = np.load(p('labels'), allow_pickle=True,encoding='latin1')
    return hgrams,labels

