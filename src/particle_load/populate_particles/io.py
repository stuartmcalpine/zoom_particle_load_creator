import numpy as np

import particle_load.mympi as mympi


def load_glass_file(num, glass_files_dir):
    """
    Load a ascii glass file.

    Parameters
    ----------
    num : int
        Number of particles in the glass file
    glass_files_dir : string
        Where are the glass files stored

    Returns
    -------
    glass : float ndarray[N,3]
        Glass particle coordinates
    """
    glass = np.loadtxt(
        glass_files_dir + "ascii_glass_%i" % num,
        dtype={"names": ["x", "y", "z"], "formats": ["f8", "f8", "f8"]},
        skiprows=1,
    )
    mympi.message("Loaded glass file, %i particles in file." % num)

    return glass
