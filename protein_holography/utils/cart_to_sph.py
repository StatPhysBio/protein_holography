import numpy as np

def cartesian_to_spherical(r: np.ndarray) -> np.ndarray:
    """
    Convert Cartesian coordinates to spherical coorindates.

    Parameters
    ----------
    r : numpy.ndarray
        Array of Cartesian coordinates x, y, z.

    Returns
    -------
    np.ndarray
        Array of spherical coordinates r_mag, theta, phi
    """
    # get cartesian coordinates                                 
    x = r[:, 0]
    y = r[:, 1]
    z = r[:, 2]

    # get spherical coords from cartesian                       
    r_mag = np.linalg.norm(r, axis=-1)
    theta = np.arccos(z / r_mag)
    phi = np.arctan2(y, x)

    # return r,theta,phi                                        
    return np.array([r_mag, theta, phi]).T

