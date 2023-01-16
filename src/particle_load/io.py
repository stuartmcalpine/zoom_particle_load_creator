import os

import h5py
import numpy as np
from scipy.io import FortranFile

import particle_load.mympi as mympi
from particle_load.parallel_functions import repartition


def _save_particle_load_as_hdf5(
    fname, coords_x, coords_y, coords_z, masses, ntot, nfile, nfile_tot, pl_params
):
    """
    Write particle load as HDF5 file for ic_gen.

    Each core writes its own file.

    Parameters
    ----------
    fname : string
        Filename
    coords_x : ndarray float[n_tot,]
        x-Coordinates
    coords_y : ndarray float[n_tot,]
        y-coordinates
    coords_z : ndarray float[n_tot,]
        z-coordinates
    masses : ndarray float[n_tot,]
        Particle masses
    n_tot : int
        Total number of particle in PL
    nfile : int
        This file part
    nfile_tot : int
        Number of files particle load is split over
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    """

    f = h5py.File(fname, "w")
    g = f.create_group("PartType1")
    g.create_dataset("Coordinates", (len(masses), 3), dtype="f8")
    g["Coordinates"][:, 0] = coords_x
    g["Coordinates"][:, 1] = coords_y
    g["Coordinates"][:, 2] = coords_z
    g.create_dataset("Masses", data=masses)
    g.create_dataset("ParticleIDs", data=np.arange(0, len(masses)))
    g = f.create_group("Header")
    g.attrs.create("nlist", len(masses))
    g.attrs.create("itot", ntot)
    g.attrs.create("nj", mympi.comm_rank)
    g.attrs.create("nfile", mympi.comm_size)
    g.attrs.create("coords", pl_params.coords / pl_params.box_size)
    g.attrs.create("radius", pl_params.radius / pl_params.box_size)
    g.attrs.create("Redshift", 1000)
    g.attrs.create("Time", 0)
    g.attrs.create("NumPart_ThisFile", [0, len(masses), 0, 0, 0])
    g.attrs.create("NumPart_Total", [0, ntot, 0, 0, 0])
    g.attrs.create("NumPart_TotalHighWord", [0, 0, 0, 0, 0])
    g.attrs.create("NumFilesPerSnapshot", mympi.comm_size)
    g.attrs.create("ThisFile", mympi.comm_rank)
    f.close()


def _save_particle_load_as_binary(
    fname, coords_x, coords_y, coords_z, masses, n_tot, nfile, nfile_tot
):
    """
    Write particle load as fortran file for ic_gen.

    Each core writes its own file.

    Parameters
    ----------
    fname : string
        Filename
    coords_x : ndarray float[n_tot,]
        x-Coordinates
    coords_y : ndarray float[n_tot,]
        y-coordinates
    coords_z : ndarray float[n_tot,]
        z-coordinates
    masses : ndarray float[n_tot,]
        Particle masses
    n_tot : int
        Total number of particle in PL
    nfile : int
        This file part
    nfile_tot : int
        Number of files particle load is split over
    """

    f = FortranFile(fname, mode="w")
    # 4+8+4+4+4 = 24
    f.write_record(
        np.int32(coords_x.shape[0]),
        np.int64(n_tot),
        np.int32(nfile),
        np.int32(nfile_tot),
        np.int32(0),
        # Now we pad the header with 6 zeros to make the header length
        # 48 bytes in total
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
        np.int32(0),
    )
    f.write_record(coords_x.astype(np.float64))
    f.write_record(coords_y.astype(np.float64))
    f.write_record(coords_z.astype(np.float64))
    f.write_record(masses.astype("float32"))
    f.close()


def _load_balance(coords_x, coords_y, coords_z, masses):
    """
    Load balance arrays between cores.

    Parameters
    ----------
    coords_x : ndarray float[n_tot,]
        x-Coordinates
    coords_y : ndarray float[n_tot,]
        y-coordinates
    coords_z : ndarray float[n_tot,]
        z-coordinates
    masses : ndarray float[n_tot,]
        Particle masses

    Returns
    -------
    coords_x, coords_y, coords_z, masses (now load balanced)
    """

    ndesired = np.zeros(mympi.comm_size, dtype=int)
    ndesired[:] = ntot / mympi.comm_size
    ndesired[-1] += ntot - sum(ndesired)
    if mympi.comm_rank == 0:
        tmp_num_per_file = ndesired[0] ** (1 / 3.0)
        print(
            "Load balancing %i particles on %i ranks (%.2f**3 per file)..."
            % (ntot, mympi.comm_size, tmp_num_per_file)
        )
        if tmp_num_per_file > pl_params.max_particles_per_ic_file ** (1 / 3.0):
            print(
                "***WARNING*** more than %s per file***"
                % (pl_params.max_particles_per_ic_file)
            )

    masses = repartition(masses, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size)
    coords_x = repartition(
        coords_x, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
    )
    coords_y = repartition(
        coords_y, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
    )
    coords_z = repartition(
        coords_z, ndesired, mympi.comm, mympi.comm_rank, mympi.comm_size
    )
    mympi.comm.barrier()
    if mympi.comm_rank == 0:
        print("Done load balancing.")

    return coords_x, coords_y, coords_z, masses


def save_pl(coords_x, coords_y, coords_z, masses, pl_params):
    """
    Save the particle load to file.

    By default this is to a Fortran file for ic_gen, but can also be saved to a
    HDF5 file.

    Parameters
    ----------
    coords_x : ndarray float[n_tot,]
        x-Coordinates
    coords_y : ndarray float[n_tot,]
        y-coordinates
    coords_z : ndarray float[n_tot,]
        z-coordinates
    masses : ndarray float[n_tot,]
        Particle masses
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    """

    # Count total number of particles in particle load.
    ntot = len(masses)
    if mympi.comm_size > 1:
        ntot = mympi.comm.allreduce(ntot)

    # Load balance arrays across cores.
    if mympi.comm_size > 1:
        coords_x, coords_y, coords_z, masses = _load_balance(
            coords_x, coords_y, coords_z, masses
        )

    assert (
        len(masses) == len(coords_x) == len(coords_y) == len(coords_z)
    ), "Array length error"

    # Make save directory.
    save_dir = f"{os.path.join(pl_params.save_dir, 'particle_load')}"
    save_dir_hdf = os.path.join(save_dir, "hdf5")
    save_dir_bin = os.path.join(save_dir, "fbinary")

    if mympi.comm_rank == 0:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        if not os.path.exists(save_dir_hdf) and pl_params.save_pl_data_hdf5:
            os.makedirs(save_dir_hdf)
        if not os.path.exists(save_dir_bin):
            os.makedirs(save_dir_bin)
    if mympi.comm_size > 1:
        mympi.comm.barrier()

    # Make sure not to save more than max_save at a time.
    max_save = 50
    lo_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[:-1]
    hi_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[1:]
    for lo, hi in zip(lo_ranks, hi_ranks):
        if mympi.comm_rank >= lo and mympi.comm_rank < hi:

            # Save HDF5 file.
            if pl_params.save_pl_data_hdf5:
                _save_particle_load_as_hdf5(
                    f"{save_dir_hdf}/PL.{mympi.comm_rank:d}.hdf5",
                    coords_x,
                    coords_y,
                    coords_z,
                    masses,
                    ntot,
                    mympi.comm_rank,
                    mympi.comm_size,
                    pl_params,
                )

            # Save to fortran binary.
            _save_particle_load_as_binary(
                f"{save_dir_bin}/PL.{mympi.comm_rank:d}",
                coords_x,
                coords_y,
                coords_z,
                masses,
                ntot,
                mympi.comm_rank,
                mympi.comm_size,
            )

            print(
                "[%i] Saved %i/%i particles..." % (mympi.comm_rank, len(masses), ntot)
            )
