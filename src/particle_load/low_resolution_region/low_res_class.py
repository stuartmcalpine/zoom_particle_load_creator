import numpy as np

import particle_load.mympi as mympi

from .find_nq import find_nq
from .plot import plot_skins


class LowResolutionRegion:
    def __init__(self, params, high_res_region):
        """
        Class that stores the information about the low-res "skin" particles
        that surround the high-res inner grid.

        They are build out symetrically from the high-res grid in ever
        decreasing resolution steps out to the edge of the full simulation
        volume.

        Parameters
        ----------
        params : dict
        high_res_region : HighResolutionRegion object
            Object that stores information about the high-res grid

        Attributes
        ----------
        side : float
            Ratio between the length of the high-res grid and the total
            simulation volume
        nq_info : dict
            Information about the best discovered value of nq
        ntot : int
            Total number of skin particles
        """

        # Case where we have a slab high-res region.
        # if pl_params.is_slab:
        #    self.compute_skins_slab()

        # Case where we have a conventional cubic high-res region.
        # else:
        self.compute_skins(params, high_res_region)

        # Report findings.
        self.print_nq_info()

        # Plot skin particles.
        if params["zoom"]["to_plot"]:
            plot_skins(self)

    def compute_skins_slab(self):
        # Starting nq is equiv of double the mass of the most massive grid particles.
        # suggested_nq = \
        #    int(num_lowest_res ** (1 / 3.) * max_cells * self.nq_mass_reduce_factor)
        # n_tot_lo = self.find_nq_slab(suggested_nq, slab_width)
        raise NotImplementedError

    def compute_skins(self, params, high_res_region):
        """
        Compute low-res skins surrounding a cubic high-res grid.

        nq is the number of particles along a side of the high-res grid in each
        skin layer. As you get further from the grid, the number of particles
        in a skin stays the same, but the volume increases, therefore the
        resolution of each paritcle lowers.

        Parameters
        ----------
        params : dict
        high_res_region : HighResolutionRegion object
            Object that stores information about the high-res grid
        """

        # Ensure boundary particles in the first skin layer wont be less
        # massive than smallest particles in the high-res grid.
        params["zoom"]["max_nq"] = np.minimum(
            int(np.floor(high_res_region.n_tot_grid_part_equiv ** (1 / 3.0))),
            params["zoom"]["_max_nq"],
        )

        # Guess starting nq
        # Starting nq is equiv of <nq_mass_reduce_factor> the mass of the most
        # massive grid particles.
        suggested_nq = np.clip(
            int(
                high_res_region.n_tot_grid_part_equiv ** (1 / 3.0)
                * params["zoom"]["nq_mass_reduce_factor"]
            ),
            params["zoom"]["min_nq"],
            params["zoom"]["max_nq"],
        )

        if mympi.comm_rank == 0:
            print(
                f"Starting: nq={suggested_nq}",
                f"(min/max bounds={params['zoom']['min_nq']}/{params['zoom']['max_nq']})",
            )

        # Compute nq. We use the starting nq as a guess, but may need to refine
        # it so that the skins (which are quantized) actually furfill the mass
        # conservation of the box).
        self.side = np.true_divide(
            high_res_region.nL_cells[0],
            params["parent"]["N_glass_L"],
        )
        self.nq_info = find_nq(self.side, suggested_nq)

        # Record total number of low res particles.
        self.n_tot = self.nq_info["n_tot_lo"]

    def print_nq_info(self):
        """Print information about the discovered value of nq"""

        for att in self.nq_info.keys():
            mympi.message(f" - nq_info: {att}: {self.nq_info[att]}")
