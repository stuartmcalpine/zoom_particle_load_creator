import matplotlib.pyplot as plt
import numpy as np

import particle_load.mympi as mympi
from particle_load.cython import get_layered_particles


def plot_skins(low_res_region):

    ntot = low_res_region.n_tot
    coords_x = np.empty(ntot, dtype="f8")
    coords_y = np.empty(ntot, dtype="f8")
    coords_z = np.empty(ntot, dtype="f8")
    masses = np.empty(ntot, dtype="f8")

    get_layered_particles(
        low_res_region.side,
        low_res_region.nq_info["nq"],
        mympi.comm_rank,
        mympi.comm_size,
        low_res_region.n_tot,
        0,
        low_res_region.nq_info["extra"],
        low_res_region.nq_info["total_volume"],
        coords_x,
        coords_y,
        coords_z,
        masses,
    )

    plt.figure(figsize=(5, 5))

    mask = np.where((coords_z > -0.000001) & (coords_z < 0.000001))
    plt.scatter(coords_x[mask], coords_y[mask], c=masses[mask], marker=".")
    plt.tight_layout(pad=0.1)
    plt.savefig("skins.png")
    plt.close()
    exit()
