import numpy as np
import os

import particle_load.mympi as mympi
import particle_load

# Where the template files are located.
_GLASS_DIR = os.path.join(particle_load.__path__[0], "glass_files")

def load_glass_file(num):
    """
    Load a ascii glass file.

    Parameters
    ----------
    num : int
        Number of particles in the glass file

    Returns
    -------
    glass : float ndarray[N,3]
        Glass particle coordinates
    """
    glass = np.loadtxt(
        os.path.join(_GLASS_DIR, f"ascii_glass_{num}"),
        dtype={"names": ["x", "y", "z"], "formats": ["f8", "f8", "f8"]},
        skiprows=1,
    )
    mympi.message("Loaded glass file, %i particles in file." % num)

    return glass
