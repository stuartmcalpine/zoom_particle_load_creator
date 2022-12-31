import numpy as np
import particle_load.mympi as mympi
from particle_load.MakeGrid import get_guess_nq


def find_nq(side, suggested_nq, eps=0.01):
    """
    Estimate what the best value of nq should be.

    Parameters
    ----------
    side : float
        Ratio between the length of the high-res grid and the total simulation volume
    suggested_nq : int
        Starting point for computing nq
    eps : float
        Tollerance level for finding a good nq

    Returns
    -------
    nq_info : dict
        Information about computed nq
    """

    nq_info = {"diff": 1.0e20}
    lbox = 1.0 / side

    # Loop over a range of potential nq's.
    for nq in np.arange(suggested_nq - 5, suggested_nq + 5, 1):
        if nq < 10:
            continue
        found_good = 0

        # Loop over a range of extras.
        for extra in range(-10, 10, 1):
            if nq + extra < 10:
                continue

            # For this nq and extra, what volume would the particles fill.
            total_volume, nlev = get_guess_nq(
                lbox, nq, extra, mympi.comm_rank, mympi.comm_size
            )
            if mympi.comm_size > 1:
                total_volume = mympi.comm.allreduce(total_volume)

            # How does this volume compare to the volume we need to fill?
            diff = np.abs(1 - (total_volume / (lbox**3.0 - 1.0**3)))

            if diff < nq_info["diff"]:
                nq_info["diff"] = diff
                nq_info["nq"] = nq
                nq_info["extra"] = extra
                nq_info["nlev"] = nlev
                nq_info["total_volume"] = total_volume

    assert nq_info["diff"] <= eps, "Did not find good nq. (diff = %.6f)" % (
        nq_info["diff"]
    )

    # Compute low res particle number for this core.
    n_tot_lo = 0
    for l in range(nq_info["nlev"]):
        if l % mympi.comm_size != mympi.comm_rank:
            continue
        if l == nq_info["nlev"] - 1:
            n_tot_lo += (nq_info["nq"] - 1 + nq_info["extra"]) ** 2 * 6 + 2
        else:
            n_tot_lo += (nq_info["nq"] - 1) ** 2 * 6 + 2
    nq_info["n_tot_lo"] = n_tot_lo

    return nq_info
