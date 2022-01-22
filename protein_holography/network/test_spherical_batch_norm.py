#
# File for testing the Batch Normalization layer
# 

import numpy as np
import tensorflow as tf

import protein_holography.network.L_batch_norm as lbn
import protein_holography.network.spherical_batch_norm as sbn

s = [2,3,1]
L_MAX = 5

sbnl = sbn.SphericalBatchNorm(0,L_MAX)


#make spherical input
input = {}
for l in range(L_MAX + 1):
    s[-1] = 2*l+1
    vars = np.array([1,10,100])
    vars = np.broadcast_to(vars[np.newaxis,:,np.newaxis],s)
    print(vars)
    input[l] = np.random.normal(loc=0.,scale=vars,size=s)

new_input = {}
for l in range(L_MAX + 1):
    new_vars = np.array([1,10,100])
    new_vars = np.broadcast_to(new_vars[np.newaxis,:,np.newaxis],s)
    print(new_vars)
    new_input[l] = np.random.normal(loc=0.,scale=new_vars,size=s)


print(input)
output = sbnl(input,True)
for i in range(1000):
    norm_out = sbnl(input,training=True)
new_norm_out = sbnl(new_input,training=False)
for l in range(L_MAX+1):
    print(np.std(new_norm_out[l],axis=(0,-1)))





print('\n Terminating successfully \n')

