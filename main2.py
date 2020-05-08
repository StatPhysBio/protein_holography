#
# Main file of program intended to run analysis on protein structural data using
# holographic machine learning techniques
#

# 
# Import statements
#
import os
import pdb_interface as pdb_int
import tensorflow as tf
import protein
from protein import aa_to_ind as aa_to_ind
import numpy as np
from clebsch import clebsch as clebsch
from tensorfieldnetworks.utils import FLOAT_TYPE

#
# parameters for the current analysis
#

# l value associated with maximum frequency used in fourier transforms
cutoffL = 1
# frequency to be used to make the holograms
k = 0.0001
# hologram radius
rH = 5.
# noise distance
d = 2.0
# directories of proteins and workoing space
casp7Dir = '/home/mpun/scratch/protein_workspace/casp7'
workDir = casp7Dir + '/workspace'
trainDir = casp7Dir + '/training30'
testDir = casp7Dir + '/validation'




#
# get train and test proteins
#
print('Getting training proteins from ' + trainDir)
trainProteins = pdb_int.get_proteins_from_dir(trainDir)
print(str(len(trainProteins)) + ' training proteins gathered')
print('Gathering testing proteins from ' + testDir)
testProteins = pdb_int.get_proteins_from_dir(testDir)
print(str(len(testProteins)) + ' testing proteins gathered')




#
# get amino acid structures from all training proteins
#
trainExamplesPerAa = 1
print('Getting ' + str(trainExamplesPerAa) + ' training holograms per amino ' +
      'acid from training proteins')
train_hgrams,train_labels = pdb_int.get_amino_acid_shapes_from_protein_list(trainProteins,trainDir,
                                                          trainExamplesPerAa,
                                                          d,rH,k,cutoffL)

#
# get amino acid holograms from all test proteins
#
testExamplesPerAa = 1
print('Getting ' + str(testExamplesPerAa) + ' testing holograms per amino ' +
      'acid from testing proteins')
test_hgrams,test_labels = pdb_int.get_amino_acid_shapes_from_protein_list(testProteins,testDir,
                                                          trainExamplesPerAa,
                                                          d,rH,k,cutoffL)

# PUT THIS IN LATER ONCE NETWORK WORKS

#
# set up the Kondor network
#
NUM_CHANNELS = pdb_int.CHANNEL_NUM
MID_LAYER_DIM = 10
NUM_CLASSES = len(protein.aa_to_ind.keys())
NUM_LAYERS = 5
# placeholder for the label
label = tf.placeholder(FLOAT_TYPE,shape=[NUM_CLASSES])

def make_cg_matrices(l_cutoff):
    # here we implement the Clebsch Gordan coefficients as 
    # 2l+1 x 2(l1)+1 x 2(l2)+1 matrices for use in taking direct products
    # Fourier coefficients
    cg_matrices = {}
    tf_cg_matrices = {}
    tf_add_cg_matrices = {}
    for l in range(l_cutoff+1):
        for l1 in range(l_cutoff+1):
            for l2 in range(0,l1+1):
                cg_matrices[(l,l1,l2)] = np.zeros([2*l + 1, 2*l1 + 1, 2*l2 +1])
                for m in range(2*l+1):
                    for m1 in range(2*l1 + 1):
                        for m2 in range(2*l2 + 1):
                            cg_matrices[(l,l1,l2)][m,m1,m2] = clebsch(l1,m1-l1,l2,m2-l2,l,m-l)
                tf_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(cg_matrices[(l,l1,l2)],dtype=tf.complex64)
                tf_add_cg_matrices[(l,l1,l2)] = tf.convert_to_tensor(np.where(cg_matrices[(l,l1,l2)]!=0,1.,0.),dtype=FLOAT_TYPE)
    return tf_cg_matrices,tf_add_cg_matrices




def get_linear_weights(dimensions,l_cutoff,layer_num):
    weights_real = {}
    weights_imag = {}
    for i in range(l_cutoff+1):
        weights_real[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                       shape=dimensions[i],
                                       name='w_real_l_'+str(i)+'_layer_'+str(layer_num))
        weights_imag[i] = tf.get_variable(dtype=FLOAT_TYPE,
                                         shape=dimensions[i],
                                         name='w_imag_l_'+str(i)+'_layer_'+str(layer_num))
    return weights_real,weights_imag

def covariant_linearity(input_real_dict,input_imag_dict,weights_real,weights_imag,l_cutoff):
    output_real = {}
    output_imag = {}
    for l in range(l_cutoff+1):
        curr_input_complex = tf.complex(input_real_dict[l],input_imag_dict[l])
        curr_weights_complex = tf.complex(weights_real[l],weights_imag[l])
        print('input shape:' +str(curr_input_complex.shape))
        print('weights shape:' +str(curr_weights_complex.shape))
        output_complex = tf.einsum('cm,ci->im',curr_input_complex,curr_weights_complex)
        output_real[l] = tf.real(output_complex)
        output_imag[l] = tf.imag(output_complex)
    return output_real,output_imag


def nonlinearity_sh_ri(input_real,input_imag,output_dim):
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
                prod_complex = tf.reshape(prod_complex,[output_dim*output_dim,2*L+1])
                prod_real = tf.real(prod_complex)
                prod_imag = tf.imag(prod_complex)

                output_real[L].append(prod_real)
                output_imag[L].append(prod_imag)
                
    for L in range(l_cutoff+1):
        output_real[L] = tf.concat(output_real[L],axis=0)     
        output_imag[L] = tf.concat(output_real[L],axis=0)
        
    return output_real,output_imag

def nonlinearity_sh_ri_all(input_real,input_imag):
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

def full_layer(input_real_dict,input_imag_dict,output_dim,layer_num):
    layer_dimensions = []
    for i in range(cutoff_l+1):
        layer_dimensions.append(input_real_dict[i].shape[0])
    layer_dimensions = [[x,output_dim] for x in layer_dimensions]
    weights_real,weights_imag = get_linear_weights(layer_dimensions,cutoff_l,layer_num)
    linear_output_real,linear_output_imag = covariant_linearity(input_real_dict,input_imag_dict,
                                                                weights_real,weights_imag,cutoff_l)
    nonlinear_output_real,nonlinear_output_imag = nonlinearity_sh_ri_all(linear_output_real,linear_output_imag)
    return nonlinear_output_real,nonlinear_output_imag

# creates tensor flow tensors for the input fourier coefficients 
def spherical_input_ri(num_channel,l_cutoff):    
    input_real_list = []
    input_imag_list = []
    for i in range(l_cutoff+1):
        input_real_list.append(tf.placeholder(FLOAT_TYPE, 
                                             shape=[num_channel,2*i+1],
                                             name='input_real_l_' + str(l)))
        input_imag_list.append(tf.placeholder(FLOAT_TYPE,
                                               shape=[num_channel,2*i+1],
                                               name='input_imag_l_' + str(l)))
    return input_real_list,input_imag_list

os.chdir('/home/mpun/scratch/protein_workspace/casp7/workspace')

tf.reset_default_graph()

# parameters for network and classification task
num_channels = 4
output_dim = 2
classes = len(aa_to_ind.keys())  
l_cutoff = 4#cutoff_l

# placeholdr for the label
label = tf.placeholder(FLOAT_TYPE,shape=[classes],name='truth_label')

# clebsch gordan matrices as part of this tensor flow graph
tf_cg_matrices,tf_add_cg_matrices = make_cg_matrices(l_cutoff)

# make inputs arrays of dimension num_channel x l_cutoff x 2l+1
inputs_real,inputs_imag = spherical_input_ri(num_channels,l_cutoff)
print(inputs_real)

# organize spherical Fourier coefficients into a dictionary
input_real_dict,input_imag_dict = make_inputs_dicts_ri(inputs_real,inputs_imag,l_cutoff)

#function call
nonlinear_output_real_0,nonlinear_output_imag_0 = full_layer(input_real_dict,input_imag_dict,
                                                             output_dim,0)

# get scalar information
scalar_output_real,scalar_output_imag = nonlinear_output_real_0[0],nonlinear_output_imag_0[0]

# concatenate real and imag output and squeeze extra dimensions
scalar_output = tf.squeeze(tf.concat((scalar_output_real,scalar_output_imag),axis=0))

print('shape = ' + str(scalar_output.shape))

# make final fully connected layer
weights = tf.get_variable(shape=[scalar_output.get_shape()[0],classes],dtype=FLOAT_TYPE,
                          initializer=tf.orthogonal_initializer(),name='weights')
biases = tf.get_variable(shape=[classes],dtype=FLOAT_TYPE,
                          initializer=tf.random_normal_initializer(mean=0.,stddev=1.0),name='biases')


boltzmann_weights = tf.einsum('ji,j->i',weights,scalar_output) + biases
loss = tf.nn.softmax_cross_entropy_with_logits(labels=label, logits=boltzmann_weights)

optim = tf.train.GradientDescentOptimizer(learning_rate=0.01)
train_op = optim.minimize(loss)
grads_and_vars = optim.compute_gradients(loss)

score = 0
print_epochs = 100
epoch = -1
max_training_score = training_examples*size
max_test_score = test_examples*size

guess_matrix = np.zeros([20,20])
loss_over_time = []
while (epoch < 1000):
    loss_ = 0.
    for i in range(len(training_labels)):
        y_ = training_labels[i]
        x_real = [training_f_coeffs_real[l][i] for l in range(cutoff_l+1)]
        x_imag = [training_f_coeffs_imag[l][i] for l in range(cutoff_l+1)]
#         x_real = [np.expand_dims(np.sum(training_f_coeffs_real[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
#         x_imag = [np.expand_dims(np.sum(training_f_coeffs_imag[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
#         print(inputs_real)
#         for i, d in zip(inputs_real,x_real):
#             print('Network shape ' +str(i.shape))
#             print('Data shape ' +str(d.shape))
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
        for i in range(len(training_labels)):
            y_ = training_labels[i]
            x_real = [training_f_coeffs_real[l][i] for l in range(cutoff_l+1)]
            x_imag = [training_f_coeffs_imag[l][i] for l in range(cutoff_l+1)]
#             x_real = [np.expand_dims(np.sum(training_f_coeffs_real[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
#             x_imag = [np.expand_dims(np.sum(training_f_coeffs_imag[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
            fd = {i: d for i, d in zip(inputs_real,x_real)}
            fd.update({i: d for i, d in zip(inputs_imag,x_imag)})
            fd[label] = y_
            boltz = sess.run(boltzmann_weights,feed_dict=fd)
            cat = list(y_).index(1)
            cat_guess = np.argmax(boltz)
            results.append((cat-cat_guess) == 0)
            guess_matrix[cat] += sp.special.softmax(boltz)
        score = np.sum(results)
        np.save('training_matrix_no_channels',guess_matrix)
        plt.imshow(np.power(guess_matrix/training_examples,0.5), cmap='hot', interpolation='nearest',
                    vmin=0.0, vmax=1.0)
        plt.xticks(range(size),[ind_to_aa[i] for i in range(size)],rotation=90)
        plt.yticks(range(size),[ind_to_aa[i] for i in range(size)])
        plt.xlabel('Predicted amino acid')
        plt.ylabel('Input amino acid')
        plt.title('Square root training prediction matrix')
        plt.colorbar()
        plt.tight_layout()
        plt.savefig('tetris_training_epoch_' + str(epoch) + '.pdf')
        plt.clf();
#        plt.show();
        print('Current train accuracy = ' + str(float(score)/max_training_score))
        if score > .70*max_training_score:
            guess_matrix = np.zeros([20,20])
            results = []
            for i in range(len(test_labels)):
                y_ = test_labels[i]
                x_real = [test_f_coeffs_real[l][i] for l in range(cutoff_l+1)]
                x_imag = [test_f_coeffs_imag[l][i] for l in range(cutoff_l+1)]
#                 x_real = [np.expand_dims(np.sum(test_f_coeffs_real[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
#                 x_imag = [np.expand_dims(np.sum(test_f_coeffs_imag[l][i],axis=0),axis=0) for l in range(cutoff_l+1)]
                fd = {i: d for i, d in zip(inputs_real,x_real)}
                fd.update({i: d for i, d in zip(inputs_imag,x_imag)})
                fd[label] = y_
                boltz = sess.run(boltzmann_weights,feed_dict=fd)
                cat = list(y_).index(1)
                cat_guess = np.argmax(boltz)
                results.append((cat-cat_guess) == 0)
                guess_matrix[cat] += sp.special.softmax(boltz)
            test_score = np.sum(results)
            print('Current test accuracy = ' + str(float(test_score)/max_test_score))
            plt.imshow(np.power(guess_matrix/training_examples,0.5), cmap='hot', interpolation='nearest',vmin=0.,vmax=1.0)
            plt.xticks(range(size),[ind_to_aa[i] for i in range(size)],rotation=90)
            plt.yticks(range(size),[ind_to_aa[i] for i in range(size)])
            plt.xlabel('Predicted amino acid')
            plt.ylabel('Input amino acid')
            plt.title('Square root test prediction matrix')
            np.save('test_matrix_no_channels',guess_matrix)
            plt.colorbar()
            plt.savefig('tetris_test_epoch_' + str(epoch) + '.pdf')
            plt.clf();
#            plt.show();

print('SUMMARY:\n Total epochs: ' + str(epoch) + ' \n Final score: ' + str(score))







print('Terminating successfully')
    

