import numpy as np

import particle_load.mympi as mympi
from particle_load.MakeGrid import get_populated_grid

from .functions import com, rescale


def _generate_uniform_grid(n_particles):
    """
    Generate a uniform grid of particles.

    Parameters
    ----------
    n_particles : int
        Total number of particles in grid to generate
        Must have a cube root

    Returns
    -------
    coords : ndarray float[n_particles,3]
        Uniform grid coordinates
    """

    if n_particles == 1:
        coords = np.ones((1, 3), dtype="f8") * 0.5
    else:
        L = int(np.rint(n_particles ** (1 / 3.0)))
        coords = np.zeros((n_particles, 3), dtype="f8")
        count = 0
        for i in range(L):
            for j in range(L):
                for k in range(L):
                    coords[count][0] = (i + 0.5) / L
                    coords[count][1] = (j + 0.5) / L
                    coords[count][2] = (k + 0.5) / L
                    count += 1

    assert np.all(coords >= 0.0) and np.all(coords <= 1.0), "Error uniform grid"
    return coords


def populate_high_res_grid(
    glass,
    coords_x,
    coords_y,
    coords_z,
    masses,
    pl_params,
    high_res_region,
    total_mass_eps=1e-6,
    com_eps=1e-6,
):
    """
    Populate the particles in the high-res grid.

    Simply, stack glass (or uniform grid) cubes beside one another.

    Type 0 grid cells will host the target resolution glass particles.
    Surrounding grid cells will be increasingly lower resolution particles until
    the high-res grid is populated.

    The layout of the grid is already determined by this point (in
    high_res_region.cell_info), this just populates the actual particles into the
    grid.

    The coords and masses arrays are populated in place.

    Parameters
    ----------
    glass : dict
        Dictionary storing the glass particles from various glass files
    coords_x : ndarray float[high_res.ntot,]
        x-coordinate array to be populated
    coords_y : ndarray float[high_res.ntot,]
        y-coordinate array to be populated
    coords_z : ndarray float[high_res.ntot,]
        z-coordinate array to be populated
    masses : ndarray float[high_res.ntot,]
        Masses array to be populated
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    high_res_region : HighResolutionRegion object
        Stores information about the high-res region
    total_mass_eps : float
        Summed particle masses in high res region tollerance
    com_eps : float
        Centre of mass tollerance
    """

    # Keep track of where we are in the array.
    cell_offset = 0

    # Loop over each cell type (i.e., particle mass) and fill up the grid.
    for i in range(len(high_res_region.cell_info["type"])):

        # Find all cells of this type.
        mask = np.where(
            high_res_region.cell_types == high_res_region.cell_info["type"][i]
        )
        assert len(mask[0]) > 0, "Dont have types that I should."

        # Glass particle coordinates.
        if high_res_region.cell_info["type"][i] == 0:
            get_populated_grid(
                high_res_region.offsets[mask],
                np.c_[
                    glass[pl_params.glass_num]["x"],
                    glass[pl_params.glass_num]["y"],
                    glass[pl_params.glass_num]["z"],
                ],
                coords_x,
                coords_y,
                coords_z,
                cell_offset,
            )

        # Grid particle coordinates.
        else:
            if pl_params.grid_also_glass:
                get_populated_grid(
                    high_res_region.offsets[mask],
                    np.c_[
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "x"
                        ],
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "y"
                        ],
                        glass[high_res_region.cell_info["num_particles_per_cell"][i]][
                            "z"
                        ],
                    ],
                    coords_x,
                    coords_y,
                    coords_z,
                    cell_offset,
                )
            else:
                get_populated_grid(
                    high_res_region.offsets[mask],
                    _generate_uniform_grid(
                        high_res_region.cell_info["num_particles_per_cell"][i]
                    ),
                    coords_x,
                    coords_y,
                    coords_z,
                    cell_offset,
                )

        # Record masses for these particles.
        masses[
            cell_offset : cell_offset
            + len(mask[0]) * high_res_region.cell_info["num_particles_per_cell"][i]
        ] = high_res_region.cell_info["particle_mass"][i]

        # Keep track of where we are in array.
        cell_offset += (
            len(mask[0]) * high_res_region.cell_info["num_particles_per_cell"][i]
        )

    # Rescale masses and coordinates of high res particles and check COM.
    max_cells = high_res_region.nL_cells[0]
    max_boxsize = high_res_region.size_mpch[0]
    coords_x[: high_res_region.n_tot] = (
        rescale(
            coords_x[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2.0,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.
    coords_y[: high_res_region.n_tot] = (
        rescale(
            coords_y[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2.0,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.
    coords_z[: high_res_region.n_tot] = (
        rescale(
            coords_z[: high_res_region.n_tot],
            -max_cells / 2.0,
            max_cells / 2.0,
            -max_boxsize / 2.0,
            max_boxsize / 2.0,
        )
        / pl_params.box_size
    )  # -0.5 > +0.5.

    # Check coords.
    assert np.all(
        np.abs(coords_x[: high_res_region.n_tot]) < 0.5
    ), "High res coords error x"
    assert np.all(
        np.abs(coords_y[: high_res_region.n_tot]) < 0.5
    ), "High res coords error y"
    assert np.all(
        np.abs(coords_z[: high_res_region.n_tot]) < 0.5
    ), "High res coords error z"

    # Check total mass.
    tot_hr_mass = np.sum(masses[: high_res_region.n_tot])
    if mympi.comm_size > 1:
        tot_hr_mass = mympi.comm.allreduce(tot_hr_mass)
    assert (
        np.abs(tot_hr_mass - (high_res_region.volume_mpch3 / pl_params.box_size**3.0))
        <= total_mass_eps
    ), "Error high res masses %.8f" % (
        np.abs(tot_hr_mass - (high_res_region.volume_mpch3 / pl_params.box_size**3.0))
    )

    # Check centre of mass.
    com_x, com_y, com_z = com(
        coords_x[: high_res_region.n_tot],
        coords_y[: high_res_region.n_tot],
        coords_z[: high_res_region.n_tot],
        masses[: high_res_region.n_tot],
    )
    if mympi.comm_rank == 0:
        print(
            "CoM for high res grid particles [%.2g %.2g %.2g]"
            % (com_x / tot_hr_mass, com_y / tot_hr_mass, com_z / tot_hr_mass)
        )
    assert com_x / tot_hr_mass <= com_eps, "Bad COM x"
    assert com_y / tot_hr_mass <= com_eps, "Bad COM y"
    assert com_z / tot_hr_mass <= com_eps, "Bad COM z"
