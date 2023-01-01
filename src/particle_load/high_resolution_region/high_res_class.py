import os
import h5py
import numpy as np

import particle_load.mympi as mympi
from particle_load.MakeGrid import *
from mpi4py import MPI

from .nearest import find_nearest_glass_file, find_nearest_cube
from .plot import plot_high_res_region


class HighResolutionRegion:
    def __init__(self, pl_params, plot=False):
        """
        Class that stores the information about the high-res grid generated
        using the HDF5 mask file.

        Parameters
        ----------
        pl_params : ParticleLoadParams
            Stores the parameters of the run
        plot : bool
            Want to plot the high-res grid?

        Attributes
        ----------
        nL_cells : ndarray - int[1,3]
            Cube root of number of cells in high-res grid (includes buffer)
        size_mpch : ndarray - float[1,3]
            Dimensions of the high-res grid in Mpc/h
        volume_mpch3 : float
            Total volume of the high-res grid in (Mpc/h)**3
        ntot_cells : int
            Total number of high-res grid cells
        offsets : ndarray int[N,3]
            Offset positions of the cells in the high-res grid
        cell_nos : ndarray int[N,3]
            Unique cell IDs of the cells in the high-res grid
        cell_types : ndarray int[N,3]
            Cell type of the cells in the high-res grid
        cell_info : dict
            For each cell type, stores the information for that type
            (ie particle mass going into them, num particles etc).

        """

        # Compute dimensions of the high-res region (in units of glass cells).
        self._set_initial_dimensions(pl_params)

        # Generate the high resolution grid.
        self._init_high_res_region(pl_params)

        # Count up the high resolution particles.
        self._count_high_res_particles(pl_params)

        # Plot the high res grid.
        if plot:
            plot_high_res_region(pl_params, self.offsets, self.cell_types)

    def _set_initial_dimensions(self, pl_params):
        """
        Compute how many target resolution glass cells are needed to fill the
        bounding box of the loaded mask as a cubic grid.

        This then dictates the size of our high-res grid in Mpc/h and glass
        cell units.

        Parameters
        ----------
        pl_params : ParticleLoadParams
            Stores the parameters of the run
        """

        # (Cube root of) number of glass cells needed to fill the high-res region.
        self.nL_cells = np.tile(
            int(
                np.ceil(
                    pl_params.high_res_region_mask.bounding_length
                    / pl_params.size_glass_cell_mpch
                )
            ),
            3,
        )

        # Want a buffer between glass cells and low-res outer shells?
        self.nL_cells += pl_params.glass_buffer_cells * 2
        assert np.all(
            self.nL_cells < pl_params.nL_glass_cells_whole_volume
        ), "To many cells in high-res region"

        # Make sure slabs do the whole box in 2 dimensions.
        # if params.data["is_slab"]:
        #    raise Exception("Test me")
        #    assert n_cells_high[0] == n_cells_high[1], "Only works for slabs in z"
        #    n_cells_high[0] = params.data["n_glass_cells"]
        #    n_cells_high[1] = params.data["n_glass_cells"]

        # Number of cells that cover bounding region.
        # params.data["radius_in_cells"] = np.true_divide(
        #    high_res_mask.params.data["bounding_length"], params.data["glass_cell_length"]
        # )

        # Make sure grid can accomodate the radius factor.
        # if params.data['is_slab == False:
        #    sys.exit('radius factor not tested')
        #    n_cells_high += \
        #        2*int(np.ceil(params.data['radius_factor*params.data['radius_in_cells-params.data['radius_in_cells))

        # What's the width of the slab in cells.
        # if params.data["is_slab"]:
        #    raise Exception("Test me")
        #    # params.data['slab_width_cells'] = \
        #    #    np.minimum(int(np.ceil(bounding_box[2] / cell_length)), n_cells)
        #    # if comm_rank == 0:
        #    #    print('Number of cells for the width of slab = %i' % params.data['slab_width_cells)

        ## Check we have a proper slab.
        # if params.data["is_slab"]:
        #    tmp_mask = np.where(n_cells_high == params.data["n_glass_cells"])[0]
        #    assert len(tmp_mask) == 2, (
        #        "For a slab simulation, 2 dimentions have to fill box "
        #        + "n_cells_high=%s n_cells=%i" % (n_cells_high, n_cells)
        #    )

        # Size of high resolution region in Mpc/h.
        self.size_mpch = [
            pl_params.size_glass_cell_mpch * float(self.nL_cells[0]),
            pl_params.size_glass_cell_mpch * float(self.nL_cells[1]),
            pl_params.size_glass_cell_mpch * float(self.nL_cells[2]),
        ]

        # Volume of high resolution region in (Mpc/h)**3
        self.volume_mpch3 = np.prod(self.size_mpch)

        # Total number of cells in high-res grid.
        self.ntot_cells = np.prod(self.nL_cells)
        assert self.ntot_cells < 2**32 / 2.0, "Total number of high res cells too big"

        # Print size of high-res grid.
        mympi.message(f"{self.nL_cells} ({self.ntot_cells}) cells in high-res grid")
        mympi.message(f"Dims of high-res grid: {self.size_mpch} Mpc/h")

    def _compute_offsets(self, pl_params):
        """
        Generate the positions of each cell in the high-res region grid.

        Cells are split between cores in MPI.

        Parameters
        ----------
        pl_params : ParticleLoadParams
            Stores the parameters of the run

        Returns
        -------
        offsets : ndarray
            The locations of the high-res region grid cells lower corners
        cell_nos : ndarray
            Unique ID of high-res region grid cell
        """

        mympi.message("Generating offsets and cell_nos...")

        # Number of high-res cells going on this core.
        this_num_cells = self.ntot_cells // mympi.comm_size
        if mympi.comm_rank < (self.ntot_cells) % mympi.comm_size:
            this_num_cells += 1
        if pl_params.verbose:
            print(f"Rank {mympi.comm_rank} gets {this_num_cells} cells.")

        # Get offsets and cell_nos for high-res cells on this core.
        offsets, cell_nos = get_grid(
            self.nL_cells[0],
            self.nL_cells[1],
            self.nL_cells[2],
            mympi.comm_rank,
            mympi.comm_size,
            this_num_cells,
        )

        # Make sure we all add up.
        check_num = len(cell_nos)
        assert (
            check_num == this_num_cells
        ), f"Error creating cells {check_num} != {this_num_cells}"

        if mympi.comm_size > 1:
            check_num = mympi.comm.allreduce(check_num)
            assert check_num == self.ntot_cells, "Error creating cells 2."

        return offsets, cell_nos

    def _add_high_res_skins(self, cell_nos, cell_types, L):
        """
        The high-res grid cells that overlap with the input mask will be filled
        with the target resolution glass cells.

        The remainder of the grid cells will be filled with lower and lower
        resolution glass (or grid) cells until we reach the edge of the high-res
        region.

        How quickly the resolution degrades away from the target resolution
        glass cells can be controlled in param file.

        This routine fills in the cell_types array, which tells us what level
        of resolution that grid cell will be filled with when we populate the
        particles. The dummy value is -1, no cell should be left with this value after
        this routine. 0 is a target resolution glass cell, which has already been
        filled in by the "get_assign_mask_cells" function earlier. Numbers greater than
        0, which we assign here, will be increasingly lower resolution.

        This routine fills the cell_types array in place.

        Parameters
        ----------
        cell_nos : array
            High-res grid cell IDs
        cell_types : array
            High-res grid cell resolutions (only type 0 filled in at this stage)
        L : int
            Number of glass cells on a side of the high resolution grid
        """

        if mympi.comm_rank == 0:
            print("Adding skins around glass cells in high res region...", end="")
        mask = np.where(cell_types == -1)
        num_to_go = len(mask[0])
        if mympi.comm_size > 1:
            num_to_go = mympi.comm.allreduce(num_to_go)
        this_type = 0
        count = 0

        # Loop until we have no more cells to fill.
        while num_to_go > 0:
            # What cells neighbour those at the current level.
            skin_cells = np.unique(
                get_find_skin_cells(cell_types, cell_nos, L, this_type)
            )

            # Share answers over MPI.
            if mympi.comm_size > 1:
                skin_cells_counts = mympi.comm.allgather(len(skin_cells))
                skin_cells_disps = np.cumsum(skin_cells_counts) - skin_cells_counts
                if mympi.comm_rank == 0:
                    skin_cells_all = np.empty(np.sum(skin_cells_counts), dtype="i4")
                else:
                    skin_cells_all = None
                mympi.comm.Gatherv(
                    skin_cells,
                    [skin_cells_all, skin_cells_counts, skin_cells_disps, MPI.INT],
                    root=0,
                )
                if mympi.comm_rank == 0:
                    skin_cells = np.unique(skin_cells_all)
                else:
                    skin_cells = None
                skin_cells = mympi.comm.bcast(skin_cells)

            # Update cell_types for cells at this level.
            idx = np.where(np.in1d(cell_nos, skin_cells))[0]
            idx2 = np.where(cell_types[idx] == -1)
            idx3 = idx[idx2]
            cell_types[idx3] = this_type + 1
            mask = np.where(cell_types == -1)
            if mympi.comm_size > 1:
                num_to_go = mympi.comm.allreduce(len(mask[0]))
            else:
                num_to_go = len(mask[0])
            this_type += 1
            count += 1

        if mympi.comm_rank == 0:
            print("added %i skins." % count)

    def _count_high_res_particles(self, pl_params):
        """
        Count total number of high-resolution particles there will be.

        Parameters
        ----------
        pl_params : ParticleLoadParams
            Stores the parameters of the run
        """

        self.cell_info = {
            "type": [],
            "num_particles_per_cell": [],
            "num_cells": [],
            "particle_mass": [],
        }
        this_tot_num_glass_particles = 0
        this_tot_num_grid_particles = 0

        # Total number of high-res cells.
        n_tot_cells = len(self.cell_types)
        if mympi.comm_size > 1:
            n_tot_cells = mympi.comm.allreduce(n_tot_cells)

        # Loop over each cell type/particle mass.
        for i in np.unique(self.cell_types):
            num_cells = len(np.where(self.cell_types == i)[0])

            self.cell_info["type"].append(i)
            this_num_cells = len(np.where(self.cell_types == i)[0])
            self.cell_info["num_cells"].append(this_num_cells)

            # Glass particles (type 0).
            if i == 0:
                self.cell_info["num_particles_per_cell"].append(pl_params.glass_num)
                this_tot_num_glass_particles += this_num_cells * pl_params.glass_num
            # Grid particles (type > 0).
            else:
                # Desired number of particles in this level of grid.
                desired_no = np.maximum(
                    pl_params.min_num_per_cell,
                    int(
                        np.ceil(
                            pl_params.glass_num
                            * np.true_divide(pl_params.skin_reduce_factor, i)
                        )
                    ),
                )

                if pl_params.grid_also_glass:
                    # Find glass file with the closest number of particles.
                    num_in_grid_cell = find_nearest_glass_file(desired_no)
                else:
                    # Find nearest cube to this number, for grid.
                    num_in_grid_cell = self.find_nearest_cube(desired_no)

                self.cell_info["num_particles_per_cell"].append(num_in_grid_cell)
                this_tot_num_grid_particles += this_num_cells * num_in_grid_cell

            # Compute the masses of the particles in each cell type.
            self.cell_info["particle_mass"].append(
                (
                    (
                        pl_params.size_glass_cell_mpch
                        / self.cell_info["num_particles_per_cell"][-1] ** (1 / 3.0)
                    )
                    / pl_params.box_size
                )
                ** 3.0
            )

        # Make them numpy arrays.
        for att in self.cell_info.keys():
            self.cell_info[att] = np.array(self.cell_info[att])

        # Check we add up.
        n_tot_cells_check = np.sum(self.cell_info["num_cells"])
        if mympi.comm_size > 1:
            n_tot_cells_check = mympi.comm.allreduce(n_tot_cells_check)
        assert n_tot_cells_check == n_tot_cells, "Bad cell count"

        # The number of glass particles if they filled the high res grid.
        self.n_tot_glass_part_equiv = pl_params.glass_num * n_tot_cells

        # How many particles in the lowest mass resolution cells in the high res grid.
        num_lowest_res = np.min(self.cell_info["num_particles_per_cell"])
        if mympi.comm_size > 1:
            num_lowest_res = mympi.comm.allreduce(num_lowest_res, op=MPI.MIN)
        self.n_tot_grid_part_equiv = n_tot_cells * num_lowest_res

        # Total number of just glass particles.
        self.tot_num_glass_particles = this_tot_num_glass_particles

        # Total number of just grid particles.
        self.tot_num_grid_particles = this_tot_num_grid_particles

        # Total number of particles in the high-res grid.
        self.n_tot = this_tot_num_grid_particles + this_tot_num_glass_particles

        # Some global properties of cells
        particle_mass_list = np.unique(self.cell_info["particle_mass"])
        if mympi.comm_size > 1:
            particle_mass_list = np.unique(
                np.concatenate(mympi.comm.allgather(particle_mass_list))
            )

        mask = particle_mass_list != np.min(particle_mass_list)
        self.cell_info["min_grid_mass"] = np.min(particle_mass_list[mask])
        self.cell_info["max_grid_mass"] = np.max(particle_mass_list[mask])
        self.cell_info["glass_mass"] = np.min(particle_mass_list)

        # Print cell properties.
        for att in self.cell_info:
            mympi.message(f" - cell_info: {att}: {self.cell_info[att]}")

    def _init_high_res_region(self, pl_params):
        """
        Make the high resolution grid.

        First it populates the cell_types array with the glass cells, using the
        HDF5 mask file. Then it populates the remaining cells in the high-res
        grid with steadily decreasing glass cells of lower resolution going
        from the mask to the edge.

        Parameters
        ----------
        pl_params : ParticleLoadParams
            Stores the parameters of the run
        """

        # Generate the grid layout.
        self.offsets, self.cell_nos = self._compute_offsets(pl_params)

        # Holds the cell types (dictates what will fill them later).
        self.cell_types = (
            np.ones(len(self.offsets), dtype="i4") * -1
        )  # Should all get overwritten.

        L = self.nL_cells[0]  # Number of glass cells on a side.
        max_boxsize = np.max(self.size_mpch)  # Length of high-res region.

        # Using a mask file.
        if pl_params.mask_file is not None:
            # Rescale mask coords into grid coords.
            mask_cell_centers = pl_params.high_res_region_mask.coords / max_boxsize * L
            mask_cell_width = (
                pl_params.high_res_region_mask.grid_cell_width / max_boxsize * L
            )
            assert np.all(np.abs(mask_cell_centers) <= L / 2.0), "Mask coords error"

            # Find out which cells in our grid will be glass given the mask.
            mympi.message("Using mask to assign glass cells...")
            get_assign_mask_cells(
                self.cell_types,
                mask_cell_centers,
                self.offsets,
                mask_cell_width,
                self.cell_nos,
            )

            # Fill the rest with degrading resolution grid or glass cells.
            self._add_high_res_skins(self.cell_nos, self.cell_types, L)

        else:
            raise Exception("Does this work without a mask?")
            # if self.is_slab:
            #    tmp_mask = np.where(np.abs(centers[:,2]) <= (self.slab_width_cells/2.)+1e-4)
            #    cell_types[tmp_mask] = 0
            # else:
            #    if comm_rank == 0: print('Computing distances to each grid cell...')
            #    dists = np.linalg.norm(centers, axis=1)
            #    if comm_rank == 0:
            #        print('Sphere rad %.2f Mpc/h center %s Mpc/h (%.2f Grid cells)'\
            #            %(self.radius, self.coords, self.radius_in_cells))
            #    mask = np.where(dists <= self.radius_factor*self.radius_in_cells)
            #    cell_types[mask] = 0
            #
            #    # Add skins around the glass cells.
            #    self.add_high_res_skins(cell_nos, cell_types, L)
