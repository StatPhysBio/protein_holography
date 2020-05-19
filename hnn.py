#
# Holographic neural network module for rotationally invariant
# holographic machine learning
#

import tensorflow as tf
import numpy as np
from tensorfieldnetworks.utils import FLOAT_TYPE
import clebsch

cutoff_l = 1

# here we implement the Clebsch Gordan coefficients as
# 2l+1 x 2(l1)+1 x 2(l2)+1 matrices for use in taking direct products
# Fourier coefficients
cg_matrices = {}
tf_cg_matrices = {}
tf_add_cg_matrices = {}
add_cg_matrices = {}
for l in range(cutoff_l + 1):
    for l1 in range(cutoff_l + 1):
        for l2 in range(0,l1+1):
            cg_matrices[(l,l1,l2)] = np.zeros([2*l + 1, 2*l1 + 1, 2*l2 +1])
            for m in range(2*l+1):
                for m1 in range(2*l1 + 1):
                    for m2 in range(2*l2 + 1):
                        cg_matrices[(l,l1,l2)][m,m1,m2] = clebsch.clebsch(l1,m1-l1,l2,m2-l2,l,m-l)
                        tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                        tf_add_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(np.where(cg_matrices[(l,l1,l2)]>0,1,0))
                        add_cg_matrices[(l,l1,l2)] = np.where(cg_matrices[(l,l1,l2)]!=0,1,0)
                    

# this function makes linear weights of given dimensions for use in taking
# linear combinations of tensorflow variables
def get_linear_weights(dimensions,max_l,layer_num):
    weights_real = {}
    weights_imag = {}
    for i in range(max_l + 1):
        weights_real[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                          shape=dimensions[i],
                                          name='w_real_l_'+str(i)+'_layer_'+str(layer_num))
        weights_imag[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                          shape=dimensions[i],
                                          name='w_imag_l_'+str(i)+'_layer_'+str(layer_num))
    return weights_real,weights_imag

# function to make inputs of the network
# creates tensor flow tensors for the input fourier coefficients
def spherical_input_ri(num_channel,l_cutoff):
    input_real_list = []
    input_imag_list = []
    for i in range(l_cutoff+1):
        input_real_list.append(tf.placeholder(FLOAT_TYPE,
                                              shape=[num_channel,2*i+1],
                                              name='input_real_l_' + str(i)))
        input_imag_list.append(tf.placeholder(FLOAT_TYPE,
                                              shape=[num_channel,2*i+1],
                                              name='input_imag_l_' + str(i)))
    return input_real_list,input_imag_list


# function that turns a list of dimension l
# into a dictionary of tf arrays indexed by l
#
# NTS: do I need to use dictionaries in the tensorflow graph?
def make_inputs_dicts_ri(input_real,input_imag,l_cutoff):
    input_real_dict = {}
    input_imag_dict = {}
    for i in range(l_cutoff+1):
        input_real_dict[i] = input_real[i]
        input_imag_dict[i] = input_imag[i]
    return input_real_dict,input_imag_dict

# linear combination of the inputs according to linear weights
def covariant_linearity(input_real_dict,input_imag_dict,weights_real,weights_imag,l_cutoff):
    output_real = {}
    output_imag = {}
    for l in range(l_cutoff+1):
        curr_input_complex = tf.complex(input_real_dict[l],input_imag_dict[l])
        curr_weights_complex = tf.complex(weights_real[l],weights_imag[l])
        output_complex = tf.einsum('cm,ci->im',curr_input_complex,curr_weights_complex)
        output_real[l] = tf.real(output_complex)
        output_imag[l] = tf.imag(output_complex)
    return output_real,output_imag

# nonlinearity CG
def nonlinearity_sh_ri_all(input_real,input_imag,l_cutoff):
    output_real = {}
    output_imag = {}

    for L in range(l_cutoff+1):
        output_real[L] = []
        output_imag[L] = []
    for l1 in range(l_cutoff+1):
        for l2 in range(l1,l_cutoff+1):
            for L in range(l2-l1,np.minimum(l_cutoff,l1+l2)+1):
                input_1_complex = tf.complex(input_real[l1],input_imag[l1])
                input_2_complex = tf.complex(input_real[l2],input_imag[l2])
                prod_complex = tf.einsum('im,Mnm,jn->ijM',input_1_complex,tf_cg_matrices[(L,l2,l1)],input_2_complex)
                dim1 = input_1_complex.shape[0]
                dim2 = input_2_complex.shape[0]
                prod_complex = tf.reshape(prod_complex,[dim1*dim2,2*L+1])
                prod_real = tf.real(prod_complex)
                prod_imag = tf.imag(prod_complex)
                
                output_real[L].append(prod_real)
                output_imag[L].append(prod_imag)
                
    for L in range(l_cutoff+1):
        output_real[L] = tf.concat(output_real[L],axis=0)
        output_imag[L] = tf.concat(output_real[L],axis=0)

    return output_real,output_imag

# a fully connected layer that takes in spherical input of dimensions {l} x c_l x m, applies a linear combination,
# a non-linear combination, and then returns an output of shape {l} x c_l' x m where c_l' is determined by
# the selection rules
def full_layer(input_real_dict,input_imag_dict,output_dim,layer_num,cutoff_l):
    layer_dimensions = []
    for i in range(cutoff_l + 1):
        layer_dimensions.append(input_real_dict[i].shape[0])
    layer_dimensions = [[x,output_dim] for x in layer_dimensions]
    print(layer_dimensions)
    weights_real,weights_imag = get_linear_weights(layer_dimensions,cutoff_l,layer_num)
    linear_output_real,linear_output_imag = covariant_linearity(input_real_dict,
                                                                input_imag_dict,
                                                                weights_real,
                                                                weights_imag,
                                                                cutoff_l)
    nonlinear_output_real,nonlinear_output_imag = nonlinearity_sh_ri_all(linear_output_real,
                                                                         linear_output_imag,
                                                                         cutoff_l)
    return nonlinear_output_real,nonlinear_output_imag


# this function produces a tensorflow graph with full CG layers
# of dimensions specified by the input dimensions
def hnn(dimensions,classes,l_cutoff):

    # placeholders for the inputs
    label = tf.placeholder(FLOAT_TYPE,shape=[classes],name='truth_label')
    inputs_real,inputs_imag = spherical_input_ri(dimensions[0],l_cutoff)

    # turn the inputs into dictionaries
    input_real_dict,input_imag_dict = make_inputs_dicts_ri(inputs_real,inputs_imag,l_cutoff)

    num_layers = len(dimensions)
    layers = []
    layers.append((input_real_dict,input_imag_dict))
    for i in range(1,num_layers):
        layers.append(full_layer(layers[i-1][0],layers[i-1][1],dimensions[i],i,l_cutoff))

    # extract scalar information from the network
    # first index is the layer number, second index is real vs imag, last index is the L order
    scalar_output_real,scalar_output_imag = layers[-1][0][0],layers[-1][1][0]

    # concatenate real and imag output and squeeze extra dimensions
    scalar_output = tf.squeeze(tf.concat((scalar_output_real,scalar_output_imag),axis=0))

    # final fully connected layer
    weights = tf.get_variable(shape=[scalar_output.get_shape()[0],classes],dtype=FLOAT_TYPE,
                              initializer=tf.orthogonal_initializer(),name='weights')
    biases = tf.get_variable(shape=[classes],dtype=FLOAT_TYPE,
                             initializer=tf.random_normal_initializer(mean=0.,stddev=1.0),name='biases')

    boltzmann_weights = tf.einsum('ji,j->i',weights,scalar_output) + biases
    loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=boltzmann_weights)


    return layers,label,inputs_real,inputs_imag,loss
        


# function to train the network
def train_on_data(training_coeffs_real,training_coeffs_imag,training_labels, #data
                  inputs_real,inputs_imag,label,sess,loss,train_op, # feed dict keys
                  epochs,print_epochs): #training parameters
    max_training_score = training_coeffs_real[0].shape[0]

    epoch = -1
    guess_matrix = np.zeros([20,20])
    loss_over_time = []
    while (epoch < epochs):
        loss_ = 0.
        for i in range(len(training_labels)):
            y_ = training_labels[i]
            x_real = [training_coeffs_real[l][i] for l in range(cutoff_l+1)]
            x_imag = [training_coeffs_imag[l][i] for l in range(cutoff_l+1)]
            fd = {i: d for i, d in zip(inputs_real,x_real)}
            fd.update({i: d for i, d in zip(inputs_imag,x_imag)})
            fd[label] = y_
            loss_val,_ = sess.run([loss,train_op],feed_dict=fd)
            loss_ += loss_val
            loss_over_time.append(loss_/float(len(training_labels)))
            epoch += 1
            results = []
            if (epoch%print_epochs) == 0:
                guess_matrix = np.zeros([20,20])
                print('Epoch ' + str(epoch) + ' loss: ' +str(loss_/len(training_labels)))
    return loss
