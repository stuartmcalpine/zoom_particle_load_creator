import matplotlib.pyplot as plt
import numpy as np
import matplotlib as mpl

import particle_load.mympi as mympi

# Set up Matplotlib
mpl.rcParams["text.usetex"] = True
mpl.rcParams["font.family"] = "serif"
mpl.rcParams["font.serif"] = "Palatino"

def plot_high_res_region(pl_params, offsets, cell_types, show_mask=False):
    """
    Have a quick look at how the high resolution region has been constructed.

    Plots a z-slice of the high-res grid cells, and loaded mask cells.

    Note the offsets are the cell corners.

    Parameters
    ----------
    pl_parms : ParticleLoadParams
        Stores the parameters of the run
    offsets : ndarray
        Positions of high-res cells
    cell_types : ndarray
        Cell type/particle mass of high-res cells
    show_mask : bool
        Also show original mask on plot ?
    """

    print(f"[Rank {mympi.comm_rank}] plotting...")

    plt.figure(figsize=(5,5))

    # Choose a z-slice.
    # Want one close to the middle of high res region.
    idx = np.abs(offsets[:, 2]).argmin()

    # Glass cells.
    mask = np.where(
        (offsets[:, 2] == offsets[:, 2][idx]) & (cell_types == 0)
    )
    plt.scatter(
        offsets[:, 0][mask] * pl_params.size_glass_cell_mpch + pl_params.size_glass_cell_mpch/2.,
        offsets[:, 1][mask] * pl_params.size_glass_cell_mpch + pl_params.size_glass_cell_mpch/2.,
        c=cell_types[mask],
        marker="x",
    )

    # Non-glass cells.
    mask = np.where(
        (offsets[:, 2] == offsets[:, 2][idx]) & (cell_types > 0)
    )
    plt.scatter(
        offsets[:, 0][mask] * pl_params.size_glass_cell_mpch + pl_params.size_glass_cell_mpch/2.,
        offsets[:, 1][mask] * pl_params.size_glass_cell_mpch + pl_params.size_glass_cell_mpch/2.,
        c=cell_types[mask],
    )

    # The original mask coordinates.
    if show_mask:
        idx2 = np.abs(pl_params.high_res_region_mask.coords[:, 2]).argmin() 
        mask = np.where(
            pl_params.high_res_region_mask.coords[:, 2]
            == pl_params.high_res_region_mask.coords[:, 2][idx2]
        )
        plt.scatter(
            pl_params.high_res_region_mask.coords[:, 0][mask],
            pl_params.high_res_region_mask.coords[:, 1][mask],
            marker=".", alpha=0.5
        )
    
    # Finish plot and save.
    plt.gca().axis("equal")
    fname = f"high_res_region_{mympi.comm_rank}.png"
    plt.xlabel(f"$x$ [$h^{{-1}}$ Mpc]")
    plt.ylabel(f"$y$ [$h^{{-1}}$ Mpc]")
    plt.tight_layout(pad=0.1)
    plt.savefig(fname)
    print(f"[Rank {mympi.comm_rank}] saved {fname}.")
    plt.close()

