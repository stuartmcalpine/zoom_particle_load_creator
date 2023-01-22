import numpy as np

import particle_load.mympi as mympi


def com(coords_x, coords_y, coords_z, masses):
    """
    Compute center of mass of a list of coords.

    Note particles may be split between cores.

    Parameters
    ----------
    coords_x : float ndarray[N,3]
        X-coordinates
    coords_y : float ndarray[N,3]
        Y-coordinates
    coords_z : float ndarray[N,3]
        Z-coordinates
    masses : float ndarray[N,1]
        Particle masses

    Returns
    -------
    com_x : float
        Centre of mass in x-direction
    com_y : float
        Centre of mass in y-direction
    com_z : float
        Centre of mass in z-direction
    """
    com_x = np.sum(coords_x * masses)
    com_y = np.sum(coords_y * masses)
    com_z = np.sum(coords_z * masses)

    if mympi.comm_size > 1:
        return (
            mympi.comm.allreduce(com_x),
            mympi.comm.allreduce(com_y),
            mympi.comm.allreduce(com_z),
        )
    else:
        return com_x, com_y, com_z


def rescale(x, x_min_old, x_max_old, x_min_new, x_max_new):
    """
    Rescale an array of numbers to a new min max.

    Parameters
    ----------
    x : float ndarray[N,]
        Array to get rescaled
    x_min_old : float
        Minimum value allowed in x
    x_max_old : float
        Maximum value allowed in x
    x_min_new : float
        Minimum value to be rescaled to
    x_max_new : float
        Maximum value to be rescaled to

    Returns
    -------
    - : float ndarray[N,]
        New array with rescaled values
    """

    assert len(x.shape) == 1, "Bad rescale array"

    return ((x_max_new - x_min_new) / (x_max_old - x_min_old)) * (
        x - x_max_old
    ) + x_max_new
