import numpy as np
from mpi4py import MPI

import particle_load.mympi as mympi
from particle_load.io import save_pl

from .functions import com, rescale
from .io import load_glass_file
from .populate_high_res import populate_high_res_grid
from .populate_low_res import populate_low_res_skins


def populate_all_particles(
    high_res_region, low_res_region, pl_params, final_tot_mass_eps=1e-6, com_eps=1e-6
):
    """
    Populate the particles into the high-res grid and the surrounding low-res skins.

    Then save the particle load.

    Parameters
    ----------
    high_res_region : HighResolutionRegion object
        Stores information about the high-res region
    low_res_region : LowResolutionRegion object
        Stores information about the low-res region
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    final_tot_mass_eps : float
        Summed particle masses tollerance
    com_eps : float
        Centre of mass tollerance
    """
    # Total number of particles in particle load.
    ntot = high_res_region.n_tot + low_res_region.n_tot

    # Initiate arrays, these will be populated in place with particles.
    coords_x = np.empty(ntot, dtype="f8")
    coords_y = np.empty(ntot, dtype="f8")
    coords_z = np.empty(ntot, dtype="f8")
    masses = np.empty(ntot, dtype="f8")

    # Load all the glass files we are going to need to populate the high res grid.
    glass = {}
    if pl_params.grid_also_glass:
        for this_glass_no in high_res_region.cell_info["num_particles_per_cell"]:
            if this_glass_no not in glass.keys():
                glass[this_glass_no] = load_glass_file(this_glass_no)
    else:
        glass[pl_params.glass_num] = load_glass_file(pl_params.glass_num)

    # Populate high resolution grid with particles.
    populate_high_res_grid(
        glass,
        coords_x,
        coords_y,
        coords_z,
        masses,
        pl_params,
        high_res_region,
    )

    # Populate low resolution skins.
    populate_low_res_skins(
        coords_x, coords_y, coords_z, masses, high_res_region, low_res_region, pl_params
    )

    # Add up total mass (should add up to 1)
    final_tot_mass = np.sum(masses)
    if mympi.comm_size > 1:
        final_tot_mass = mympi.comm.allreduce(final_tot_mass)

    # Make sure total mass is 1.
    tmp_tol = np.abs(1 - final_tot_mass)
    assert tmp_tol <= final_tot_mass_eps, "Final mass error %.8f != 0.0" % tmp_tol

    # Check centre of mass of all particles.
    com_x, com_y, com_z = com(coords_x, coords_y, coords_z, masses)
    if mympi.comm_rank == 0:
        print(
            "CoM for all particles [%.2g %.2g %.2g]"
            % (com_x / final_tot_mass, com_y / final_tot_mass, com_z / final_tot_mass)
        )
    assert com_x / final_tot_mass <= com_eps, "Bad COM x"
    assert com_y / final_tot_mass <= com_eps, "Bad COM y"
    assert com_z / final_tot_mass <= com_eps, "Bad COM z"

    # Wrap coords to chosen center.
    wrap_coords = rescale(pl_params.coords, 0, pl_params.box_size, 0, 1.0)
    coords_x = np.mod(coords_x + wrap_coords[0] + 1.0, 1.0)
    coords_y = np.mod(coords_y + wrap_coords[1] + 1.0, 1.0)
    coords_z = np.mod(coords_z + wrap_coords[2] + 1.0, 1.0)

    # Check coords and masses.
    assert np.all(coords_x > 0) and np.all(coords_x < 1.0), "Coords x wrap error"
    assert np.all(coords_y > 0) and np.all(coords_y < 1.0), "Coords y wrap error"
    assert np.all(coords_z > 0) and np.all(coords_z < 1.0), "Coords z wrap error"
    assert np.all(masses > 0.0) and np.all(masses < 1.0), "Mass number error"

    # Save the particle load.
    save_pl(coords_x, coords_y, coords_z, masses, pl_params)
