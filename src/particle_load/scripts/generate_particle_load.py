import argparse

import numpy as np

import particle_load.mympi as mympi
from particle_load.high_resolution_region import HighResolutionRegion
from particle_load.ic_gen_functions import compute_fft_stats
from particle_load.low_resolution_region import LowResolutionRegion

from particle_load.make_param_files import build_param_dict
from particle_load.make_param_files import make_ic_param_files, make_swift_param_files 
from particle_load.params import ParticleLoadParams
from particle_load.populate_particles import populate_all_particles

def main():
    # Command line args.
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-IC", "--make_ic_gen_param_files", help="Make ic_gen files.", action="store_true"
    )
    parser.add_argument("param_file", help="Parameter file.")
    parser.add_argument(
        "-S", "--make_swift_param_files", help="Make SWIFT files.", action="store_true"
    )
    parser.add_argument(
        "-PL", "--save_pl_data", help="Save fortran PL files.", action="store_true"
    )
    parser.add_argument(
        "-PL_HDF5", "--save_pl_data_hdf5", help="Save HDF5 PL files.", action="store_true"
    )
    parser.add_argument("-MPI", "--with_mpi", help="Run over MPI.", action="store_true")
    args = parser.parse_args()

    # Init MPI.
    mympi.init_mpi(args.with_mpi)

    # Read the parameter file.
    if mympi.comm_rank == 0:
        pl_params = ParticleLoadParams(args)
    else:
        pl_params = None
    if mympi.comm_size > 1:
        pl_params = mympi.comm.bcast(pl_params, root=0)

    # Print stats.
    def print_stats(high_res_region, low_res_region, pl_params, n_tot):
        mympi.print_section_header("Totals")

        min_ranks = np.true_divide(n_tot, pl_params.max_particles_per_ic_file)

        mympi.message(f"Total number of particles {n_tot} ({n_tot**(1/3.):.1f} cubed)")
        if high_res_region is not None:
            if mympi.comm_size > 1:
                frac_glass = mympi.comm.allreduce(high_res_region.tot_num_glass_particles) / n_tot
                frac_grid = mympi.comm.allreduce(high_res_region.tot_num_grid_particles) / n_tot
            else:
                frac_glass = high_res_region.tot_num_glass_particles / n_tot
                frac_grid = high_res_region.tot_num_grid_particles / n_tot
            mympi.message(f" - Fraction that are glass particles = {frac_glass*100.:.3f}%")
            mympi.message(f" - Fraction that are grid particles = {frac_grid*100.:.3f}%")
        mympi.message(
            f"Num ranks needed for less than <max_particles_per_ic_file> = {min_ranks:.2f}"
        )


    # Go...

    # Generate zoom-region particle load.
    if pl_params.is_zoom:

        # High resolution grid.
        mympi.print_section_header("High resolution grid")
        high_res_region = HighResolutionRegion(pl_params)

        # Low resolution boundary particles.
        mympi.print_section_header("Low resolution shells")
        low_res_region = LowResolutionRegion(pl_params, high_res_region)

        # Total number of particles in particle load.
        n_tot_local = high_res_region.n_tot + low_res_region.n_tot
        if mympi.comm_size > 1:
            n_tot = mympi.comm.allreduce(n_tot_local)
        else:
            n_tot = n_tot_local

        # Compute FFT size.
        mympi.print_section_header("FFT stats")
        if mympi.comm_rank == 0:
            compute_fft_stats(np.max(high_res_region.size_mpch), n_tot, pl_params)

        # Populate the grid with particles.
        if pl_params.save_pl_data:
            mympi.print_section_header("Populating and saving particles")
            populate_all_particles(high_res_region, low_res_region, pl_params)

    ## Generate particle load for uniform volume.
    # else:
    #    high_res_region = None
    #    low_res_region = None
    #
    #    # Total number of particles in particle load.
    #    n_tot = pl_params.n_particles
    #    min_ranks = np.true_divide(n_tot, pl_params.max_particles_per_ic_file)
    #
    #    # Compute FFT size.
    #    mympi.print_section_header("FFT stats")
    #    if mympi.comm_rank == 0:
    #        compute_fft_stats(None, None, n_tot, pl_params)
    #
    # Print total number of particles in particle load.
    print_stats(high_res_region, low_res_region, pl_params, n_tot)

    # Make the param files.
    if mympi.comm_rank == 0:
        param_dict = build_param_dict(pl_params, high_res_region)

        if pl_params.make_ic_gen_param_files:
            make_ic_param_files(param_dict)

        if pl_params.make_swift_param_files:
            make_swift_param_files(param_dict)
