import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import particle_load.mympi as mympi


def plot_high_res_region(params, offsets, cell_types, show_mask=False):
    """
    Have a quick look at how the high resolution region has been constructed.

    Plots a z-slice of the high-res grid cells, and loaded mask cells.

    Note the offsets are the (negative most) cell corners not cell centres.

    Parameters
    ----------
    params : dict
    offsets : ndarray
        Positions of high-res cells
    cell_types : ndarray
        Cell type/particle mass of high-res cells
    show_mask : bool
        Also show original mask on plot ?
    """

    print(f"[Rank {mympi.comm_rank}] plotting...")

    plt.figure(figsize=(5, 5))

    # Choose a z-slice.
    # Want one close to the middle of high res region.
    idx = np.abs(offsets[:, 2]).argmin()

    L_glass = params["glass_file"]["L_mpch"]
    coords = params["zoom"]["mask_coordinates"]

    # Glass cells.
    mask = np.where((offsets[:, 2] == offsets[:, 2][idx]) & (cell_types == 0))
    plt.scatter(
        offsets[:, 0][mask] * L_glass + L_glass / 2.0,
        offsets[:, 1][mask] * L_glass + L_glass / 2.0,
        c=cell_types[mask],
        marker="x",
    )

    # Non-glass cells.
    mask = np.where((offsets[:, 2] == offsets[:, 2][idx]) & (cell_types > 0))
    plt.scatter(
        offsets[:, 0][mask] * L_glass + L_glass / 2.0,
        offsets[:, 1][mask] * L_glass + L_glass / 2.0,
        c=cell_types[mask],
    )

    # The original mask coordinates.
    if show_mask:
        idx2 = np.abs(coords[:, 2]).argmin()
        mask = np.where(coords[:, 2] == coords[:, 2][idx2])
        plt.scatter(
            coords[:, 0][mask],
            coords[:, 1][mask],
            marker=".",
            alpha=0.5,
        )

    # Finish plot and save.
    plt.gca().axis("equal")
    fname = f"high_res_region_{mympi.comm_rank}.png"
    plt.xlabel(r"$x$ [Mpc/h]")
    plt.ylabel(r"$y$ [Mpc/h]")
    plt.tight_layout(pad=0.1)
    plt.savefig(fname)
    print(f"[Rank {mympi.comm_rank}] saved {fname}.")
    plt.close()
