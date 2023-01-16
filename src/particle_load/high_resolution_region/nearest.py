import os

import numpy as np

import particle_load

# Where the template files are located.
_GLASS_DIR = os.path.join(particle_load.__path__[0], "glass_files")


def find_nearest_glass_file(num):
    """
    Find glass file that has the closest number of particles to <num>.

    Parameters
    ----------
    num : int
        Number of particles we want to find the closest match for

    Returns
    -------
    files[idx] : string
        The filename of the glass file with the closest match
    """
    files = os.listdir(_GLASS_DIR)

    files = np.array(
        [int(x.split("_")[2]) for x in files if "ascii_glass_" in x], dtype="i8"
    )
    idx = np.abs(files - num).argmin()

    return files[idx]


def find_nearest_cube(num):
    """
    Find the nearest number to <num> that has a cube root.

    Need our "grid" cells to be cube roots.

    Parameters
    ----------
    num : int
        Number of particles we want to find the closest match for

    Returns
    -------
        Closest number to <num> that has a cube root
    """
    return int(np.ceil(num ** (1 / 3.0)) ** 3.0)
