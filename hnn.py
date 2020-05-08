#
# Holographic neural network module for rotationally invariant
# holographic machine learning
#

def get_linear_weights(dimensions,max_l,layer_num):
    weights_real = {}
    weights_imag = {}
    for i in range(max_l + 1):
        weights_real[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                          shape=dimensions,
                                          name='w_real_l_'+str(i)+'_layer_'+str(layer_num))
        weights_imag[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                          shape=dimensions,
                                          name='w_imag_l_'+str(i)+'_layer_'+str(layer_num))
    return weights_real,weights_imag

