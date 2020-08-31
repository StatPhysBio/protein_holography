import numpy as np
import tensorflow as tf
import numpy as np
import scipy

def get_dataset(hologram_dir, examples_per_aa, k, d, l_max):
    p = lambda part: "{}//train_{}_examplesPerAA={}_k={}_d={}_l={}.npy".format(
        hologram_dir, part, examples_per_aa, k, d, l_max)

    hgrams_real = np.load(p('hgram_real_example'), allow_pickle=True,encoding='latin1')[()]
    hgrams_imag = np.load(p('hgram_imag_example'), allow_pickle=True,encoding='latin1')[()]
    hgrams = {}
    for l in range(l_max + 1):
        hgrams[l] = (hgrams_real[l] + 1j * hgrams_imag[l]).astype("complex64")

    #print(hgrams[1][0])
    labels = np.load(p('labels'), allow_pickle=True,encoding='latin1')
    return tf.data.Dataset.from_tensor_slices((hgrams,labels))

if __name__ == "__main__":
    ds = get_dataset('../holograms', 1000, 0.0001, 10.0, 6)
    print(ds.batch(1).batch(2).take(2))