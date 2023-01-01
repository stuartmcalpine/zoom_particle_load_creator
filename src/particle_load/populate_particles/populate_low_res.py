import numpy as np

import particle_load.mympi as mympi
from particle_load.cython import get_layered_particles


def populate_low_res_skins(
    coords_x, coords_y, coords_z, masses, high_res_region, low_res_region, pl_params
):
    """
    Populate particles that live in the low-resolution skins.

    Coordinate and mass arrays are populated in place.

    Parameters
    ----------
    coords_x : ndarray float[high_res.ntot,]
        x-coordinate array to be populated
    coords_y : ndarray float[high_res.ntot,]
        y-coordinate array to be populated
    coords_z : ndarray float[high_res.ntot,]
        z-coordinate array to be populated
    masses : ndarray float[high_res.ntot,]
        Masses array to be populated
    high_res_region : HighResolutionRegion object
        Stores information about the high-res region
    low_res_region : LowResolutionRegion object
        Stores information about the low-res region
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    """

    # Generate outer particles of low res grid with growing skins.
    if pl_params.is_slab:
        raise NotImplementedError
    #   if comm_rank == 0:
    #     print('Putting low res particles around slab of width %.2f Mpc/h' % \
    #                  min_boxsize)
    #        if n_tot_lo > 0:
    #            get_layered_particles_slab(min_boxsize, self.box_size,
    #                self.nq_info['starting_nq'], self.nq_info['nlev_slab'],
    #                self.nq_info['dv_slab'], comm_rank, comm_size, n_tot_lo, n_tot_hi,
    #                coords_x, coords_y, coords_z, masses, self.nq_info['nq_reduce'],
    #                self.nq_info['extra'])
    else:
        get_layered_particles(
            low_res_region.side,
            low_res_region.nq_info["nq"],
            mympi.comm_rank,
            mympi.comm_size,
            low_res_region.n_tot,
            high_res_region.n_tot,
            low_res_region.nq_info["extra"],
            low_res_region.nq_info["total_volume"],
            coords_x,
            coords_y,
            coords_z,
            masses,
        )

    # Check coordinates.
    assert np.all(
        np.abs(coords_x[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords x error"
    assert np.all(
        np.abs(coords_y[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords y error"
    assert np.all(
        np.abs(coords_z[high_res_region.n_tot :]) <= 0.5
    ), "Low res coords z error"
