#
# Moduel to implement geomtric transformations used in parsing pdb files
#

import numpy as np

def cartesian_to_spherical(r):
    # get cartesian coordinates
    x = r[0]
    y = r[1]
    z = r[2]

    # get spherical coords from cartesian
    r_mag = np.sqrt(np.sum([x_*x_ for x_ in r]))
    t = np.arccos(z/r_mag)
    p = np.arctan2(y,x)

    # return r,theta,phi
    return r_mag,t,p
    
