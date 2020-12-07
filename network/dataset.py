import numpy as np
import tensorflow as tf
import numpy as np
import scipy

def get_dataset(hologram_dir, ch, examples_per_aa, l_max, k, d, rH, aas):
    p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_rH={}.npy".format(
        hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH)
    if len(aas) > 0:
        p = lambda part: "{}/{}_ch={}_e={}_l={}_k={}_d={}_rH={}_aas={}.npy".format(
            hologram_dir, part, ch, examples_per_aa, l_max, k, d, rH, aas)


    hgrams_real = np.load(p('hgram_real'), allow_pickle=True,encoding='latin1')[()]
    hgrams_imag = np.load(p('hgram_imag'), allow_pickle=True,encoding='latin1')[()]
    hgrams = {}
    for l in range(l_max + 1):
        hgrams[l] = (hgrams_real[l] + 1j * hgrams_imag[l]).astype("complex64")

    #print(hgrams[1][0])
    labels = np.load(p('labels'), allow_pickle=True,encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((hgrams,labels))

if __name__ == "__main__":
    ds = get_dataset('../holograms', 1000, 0.0001, 10.0, 6)
    print(ds.batch(1).batch(2).take(2))


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
