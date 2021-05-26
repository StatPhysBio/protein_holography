#
# Python module for implementing creating the Carbon-Oxygen-alpha Carbon 
# coordinate system 
#
import numpy as np
# Levi-Civita tensor for taking cross product
def get_eijk():
    """
    Constant Levi-Civita tensor

    Returns:
        tf.Tensor of shape [3, 3, 3]
    """
    eijk_ = np.zeros((3, 3, 3))
    eijk_[0, 1, 2] = eijk_[1, 2, 0] = eijk_[2, 0, 1] = 1.
    eijk_[0, 2, 1] = eijk_[2, 1, 0] = eijk_[1, 0, 2] = -1.
    return eijk_

def get_COA_axes(res):
    
    # check to make sure that all atoms necessary for the axes ar present
    if 'O' not in res:
        print('Error: No Oxygen found')
    if 'CA' not in res:
        print('Error: No alpha Carbon found')
    if 'C' not in res:
        print('Error: No Carbon found')

    # get coordinate system based on atom positions
    CCA_dir = res['C'].get_coord() - res['CA'].get_coord()
    CCA_dir_mag = np.sum([x_*x_ for x_ in CCA_dir])
    x_hat = (1./np.sqrt(CCA_dir_mag))*CCA_dir
    
    OCA_dir = res['O'].get_coord() - res['CA'].get_coord()
    y_dir = OCA_dir - np.einsum('i,i',OCA_dir,x_hat)*x_hat
    y_dir_mag = np.sum([x_*x_ for x_ in y_dir])
    y_hat = (1./np.sqrt(y_dir_mag))*y_dir
    
    z_hat = np.einsum('ijk,j,k->i',get_eijk(),x_hat,y_hat)

    return np.array([x_hat,y_hat,z_hat])

def get_ACN_axes(res):
    
    # check to make sure that all atoms necessary for the axes ar present
    if 'N' not in res:
        print('Error: No Nitrogen found')
    if 'CA' not in res:
        print('Error: No alpha Carbon found')
    if 'C' not in res:
        print('Error: No Carbon found')

    # get coordinate system based on atom positions
    CCA_dir = res['CA'].get_coord() - res['C'].get_coord()
    CCA_dir_mag = np.sum([x_*x_ for x_ in CCA_dir])
    x_hat = (1./np.sqrt(CCA_dir_mag))*CCA_dir
    
    OCA_dir = res['CA'].get_coord() - res['N'].get_coord()
    y_dir = OCA_dir - np.einsum('i,i',OCA_dir,x_hat)*x_hat
    y_dir_mag = np.sum([x_*x_ for x_ in y_dir])
    y_hat = (1./np.sqrt(y_dir_mag))*y_dir
    
    z_hat = np.einsum('ijk,j,k->i',get_eijk(),x_hat,y_hat)

    return np.array([x_hat,y_hat,z_hat])
