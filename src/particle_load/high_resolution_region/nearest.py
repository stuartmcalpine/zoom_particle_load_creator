import numpy as np
import os

def find_nearest_glass_file(num, glass_files_dir):
    """
    Find glass file that has the closest number of particles to <num>.
    
    Parameters
    ----------
    num : int
        Number of particles we want to find the closest match for
    glass_files_dir : string
        Directory containing the ascii glass files

    Returns
    -------
    files[idx] : string
        The filename of the glass file with the closest match
    """
    files = os.listdir(glass_files_dir)

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
