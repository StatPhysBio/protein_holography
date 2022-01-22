#
# File for testing the Batch Normalization layer
# 

import numpy as np
import tensorflow as tf

import protein_holography.network.L_batch_norm as lbn

s = [100,3,100]

bnl = lbn.LBatchNorm()
#print(bnl.__dict__)
bnl.build(s)

print('\nMoving mean')
print(bnl.moving_mean)

print('\nMoving Variance')
print(bnl.moving_variance)

input = []
vars = np.array([1,10,1000])
vars = np.broadcast_to(vars[np.newaxis,:,np.newaxis],s)
print(vars)
input = np.random.normal(loc=0.,scale=vars,size=s)

new_vars = np.array([1,1,1])
new_vars = np.broadcast_to(new_vars[np.newaxis,:,np.newaxis],s)
print(new_vars)
new_input = np.random.normal(loc=0.,scale=new_vars,size=s)


print(input)
print('\n\nInput shape: {}'.format(input.shape))
for i in range(100):
    print('Round {}'.format(i))
    norm_out = bnl(input,training=True)
new_norm_out = bnl(new_input,training=False)
print(np.var(new_norm_out,axis=(0,-1)))





print('\n Terminating successfully \n')

