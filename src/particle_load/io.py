import os
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional, Union

import h5py
import numpy as np
from scipy.io import FortranFile

import particle_load.mympi as mympi
from particle_load.parallel_functions import repartition


def validate_params(params: Dict[str, Any], required_keys: Dict[str, List[str]]) -> None:
    """
    Validate that all required keys exist in the params dictionary.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing parameters
    required_keys : Dict[str, List[str]]
        Dictionary mapping section names to lists of required keys
        
    Raises
    ------
    KeyError
        If a required key is missing
    """
    for section, keys in required_keys.items():
        if section not in params:
            raise KeyError(f"Missing required section: {section}")
        
        for key in keys:
            if key not in params[section]:
                raise KeyError(f"Missing required parameter: {section}.{key}")


def validate_arrays(coords_x: np.ndarray, coords_y: np.ndarray, 
                   coords_z: np.ndarray, masses: np.ndarray) -> None:
    """
    Validate input arrays for particle load operations.
    
    Parameters
    ----------
    coords_x : np.ndarray
        x-Coordinates
    coords_y : np.ndarray
        y-Coordinates
    coords_z : np.ndarray
        z-Coordinates
    masses : np.ndarray
        Particle masses
        
    Raises
    ------
    ValueError
        If arrays are empty or have different lengths
    """
    if len(masses) == 0:
        raise ValueError("Particle arrays cannot be empty")
    
    if not (len(masses) == len(coords_x) == len(coords_y) == len(coords_z)):
        raise ValueError(
            f"Array length mismatch: masses({len(masses)}), coords_x({len(coords_x)}), "
            f"coords_y({len(coords_y)}), coords_z({len(coords_z)})"
        )


def _save_particle_load_as_hdf5(
    fname: str,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    masses: np.ndarray,
    ntot: int,
    nfile: int,
    nfile_tot: int,
    params: Dict[str, Any]
) -> None:
    """
    Write particle load as HDF5 file for ic_gen.

    Each core writes its own file.

    Parameters
    ----------
    fname : str
        Filename
    coords_x : np.ndarray
        x-Coordinates
    coords_y : np.ndarray
        y-Coordinates
    coords_z : np.ndarray
        z-Coordinates
    masses : np.ndarray
        Particle masses
    ntot : int
        Total number of particles in PL
    nfile : int
        This file part
    nfile_tot : int
        Number of files particle load is split over
    params : Dict[str, Any]
        Dictionary containing simulation parameters
        Required keys:
        - parent.coords: Coordinate system
        - parent.box_size: Box size
    """
    # Validate parameters
    required_keys = {"parent": ["coords", "box_size"]}
    validate_params(params, required_keys)
    
    # Validate arrays
    validate_arrays(coords_x, coords_y, coords_z, masses)
    
    try:
        with h5py.File(fname, "w") as f:
            # Create and populate particle data group
            g = f.create_group("PartType1")
            g.create_dataset("Coordinates", (len(masses), 3), dtype="f8")
            g["Coordinates"][:, 0] = coords_x
            g["Coordinates"][:, 1] = coords_y
            g["Coordinates"][:, 2] = coords_z
            g.create_dataset("Masses", data=masses)
            g.create_dataset("ParticleIDs", data=np.arange(0, len(masses)))
            
            # Create and populate header group
            g = f.create_group("Header")
            g.attrs.create("nlist", len(masses))
            g.attrs.create("itot", ntot)
            g.attrs.create("nj", mympi.comm_rank)
            g.attrs.create("nfile", mympi.comm_size)
            g.attrs.create("coords", params["parent"]["coords"] / params["parent"]["box_size"])
            g.attrs.create("Redshift", 1000)
            g.attrs.create("Time", 0)
            g.attrs.create("NumPart_ThisFile", [0, len(masses), 0, 0, 0])
            g.attrs.create("NumPart_Total", [0, ntot, 0, 0, 0])
            g.attrs.create("NumPart_TotalHighWord", [0, 0, 0, 0, 0])
            g.attrs.create("NumFilesPerSnapshot", mympi.comm_size)
            g.attrs.create("ThisFile", mympi.comm_rank)
    except Exception as e:
        print(f"Error writing HDF5 file {fname}: {str(e)}")
        raise


def _save_particle_load_as_binary(
    fname: str,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    masses: np.ndarray,
    ntot: int,
    nfile: int,
    nfile_tot: int
) -> None:
    """
    Write particle load as fortran file for ic_gen.

    Each core writes its own file.

    Parameters
    ----------
    fname : str
        Filename
    coords_x : np.ndarray
        x-Coordinates
    coords_y : np.ndarray
        y-Coordinates
    coords_z : np.ndarray
        z-Coordinates
    masses : np.ndarray
        Particle masses
    ntot : int
        Total number of particles in PL
    nfile : int
        This file part
    nfile_tot : int
        Number of files particle load is split over
    """
    # Validate arrays
    validate_arrays(coords_x, coords_y, coords_z, masses)
    
    try:
        with FortranFile(fname, mode="w") as f:
            # 4+8+4+4+4 = 24
            f.write_record(
                np.int32(coords_x.shape[0]),
                np.int64(ntot),
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
    except Exception as e:
        print(f"Error writing binary file {fname}: {str(e)}")
        raise


def _load_balance(
    ntot: int,
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    masses: np.ndarray,
    max_particles_per_ic_file: int
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Load balance arrays between cores.

    Parameters
    ----------
    ntot : int
        Total number of particles
    coords_x : np.ndarray
        x-Coordinates
    coords_y : np.ndarray
        y-Coordinates
    coords_z : np.ndarray
        z-Coordinates
    masses : np.ndarray
        Particle masses
    max_particles_per_ic_file : int
        Max number of particles we want per core

    Returns
    -------
    Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        coords_x, coords_y, coords_z, masses (now load balanced)
    """
    # Input validation
    if ntot <= 0:
        raise ValueError("Total number of particles must be positive")
    if max_particles_per_ic_file <= 0:
        raise ValueError("Maximum particles per file must be positive")
    
    validate_arrays(coords_x, coords_y, coords_z, masses)
    
    # Calculate desired distribution of particles across ranks
    ndesired = np.zeros(mympi.comm_size, dtype=int)
    # Use floor division and then handle remainder to avoid floating-point division issues
    particles_per_rank = ntot // mympi.comm_size
    remainder = ntot % mympi.comm_size
    
    ndesired[:] = particles_per_rank
    # Distribute remainder to last rank
    ndesired[-1] += remainder
    
    if mympi.comm_rank == 0:
        particles_per_file = max(ndesired)
        tmp_num_per_file = particles_per_file ** (1 / 3.0)
        print(
            f"Load balancing {ntot} particles on {mympi.comm_size} ranks "
            f"(max {tmp_num_per_file:.2f}**3 per file)..."
        )
        if tmp_num_per_file > max_particles_per_ic_file ** (1 / 3.0):
            print(
                f"***WARNING*** more than {max_particles_per_ic_file} per file***"
            )

    # Repartition data across ranks
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


def save_pl(
    coords_x: np.ndarray,
    coords_y: np.ndarray,
    coords_z: np.ndarray,
    masses: np.ndarray,
    params: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Save the particle load to file.

    By default this is to a Fortran file for ic_gen, but can also be saved to a
    HDF5 file.

    Parameters
    ----------
    coords_x : np.ndarray
        x-Coordinates
    coords_y : np.ndarray
        y-Coordinates
    coords_z : np.ndarray
        z-Coordinates
    masses : np.ndarray
        Particle masses
    params : Dict[str, Any]
        Dictionary containing simulation parameters
        Required keys:
        - output.path: Output directory path
        - ic_gen.max_particles_per_ic_file: Maximum particles per file
        - cmd_args.save_pl_data_hdf5: Whether to save HDF5 files
        - parent.coords: Coordinate system
        - parent.box_size: Box size

    Returns
    -------
    Dict[str, Any]
        Updated params dictionary with information about saved files
    """
    # Validate required parameters
    required_keys = {
        "output": ["path"],
        "ic_gen": ["max_particles_per_ic_file"],
        "cmd_args": ["save_pl_data_hdf5"],
        "parent": ["coords", "box_size"]
    }
    validate_params(params, required_keys)
    
    # Validate input arrays
    validate_arrays(coords_x, coords_y, coords_z, masses)
    
    # Make a deep copy to avoid modifying the original
    params = params.copy()
    
    # Count total number of particles in particle load.
    local_count = len(masses)
    ntot = local_count
    if mympi.comm_size > 1:
        ntot = mympi.comm.allreduce(local_count)

    # Load balance arrays across cores.
    if mympi.comm_size > 1:
        coords_x, coords_y, coords_z, masses = _load_balance(
            ntot,
            coords_x,
            coords_y,
            coords_z,
            masses,
            params["ic_gen"]["max_particles_per_ic_file"],
        )

    # Verify arrays have consistent lengths
    if not (len(masses) == len(coords_x) == len(coords_y) == len(coords_z)):
        raise ValueError(
            f"Array length mismatch after load balancing: "
            f"masses({len(masses)}), coords_x({len(coords_x)}), "
            f"coords_y({len(coords_y)}), coords_z({len(coords_z)})"
        )

    # Make save directory using pathlib for better path handling
    save_dir = Path(params['output']['path']) / 'particle_load'
    save_dir_hdf = save_dir / 'hdf5'
    save_dir_bin = save_dir / 'fbinary'

    # Create directories on rank 0
    if mympi.comm_rank == 0:
        save_dir.mkdir(parents=True, exist_ok=True)
        save_dir_bin.mkdir(parents=True, exist_ok=True)
        if params["cmd_args"]["save_pl_data_hdf5"]:
            save_dir_hdf.mkdir(parents=True, exist_ok=True)
    
    # Wait for rank 0 to create directories
    if mympi.comm_size > 1:
        mympi.comm.barrier()

    # Make sure not to save more than max_save at a time.
    max_save = 50
    lo_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[:-1]
    hi_ranks = np.arange(0, mympi.comm_size + max_save, max_save)[1:]
    
    # Track files saved by this rank
    saved_files = []
    
    for lo, hi in zip(lo_ranks, hi_ranks):
        if mympi.comm_rank >= lo and mympi.comm_rank < hi:
            # Save HDF5 file if requested
            if params["cmd_args"]["save_pl_data_hdf5"]:
                hdf5_filename = save_dir_hdf / f"PL.{mympi.comm_rank:d}.hdf5"
                try:
                    _save_particle_load_as_hdf5(
                        str(hdf5_filename),
                        coords_x,
                        coords_y,
                        coords_z,
                        masses,
                        ntot,
                        mympi.comm_rank,
                        mympi.comm_size,
                        params,
                    )
                    saved_files.append(str(hdf5_filename))
                except Exception as e:
                    print(f"Error saving HDF5 file on rank {mympi.comm_rank}: {str(e)}")

            # Save to fortran binary
            binary_filename = save_dir_bin / f"PL.{mympi.comm_rank:d}"
            try:
                _save_particle_load_as_binary(
                    str(binary_filename),
                    coords_x,
                    coords_y,
                    coords_z,
                    masses,
                    ntot,
                    mympi.comm_rank,
                    mympi.comm_size,
                )
                saved_files.append(str(binary_filename))
            except Exception as e:
                print(f"Error saving binary file on rank {mympi.comm_rank}: {str(e)}")

            print(
                f"[{mympi.comm_rank}] Saved {len(masses)}/{ntot} particles..."
            )
    
    # Wait for all ranks to finish saving
    if mympi.comm_size > 1:
        mympi.comm.barrier()
    
    # Store information about saved files in params
    if "saved_files" not in params:
        params["saved_files"] = {}
    
    params["saved_files"]["rank"] = mympi.comm_rank
    params["saved_files"]["total_particles"] = ntot
    params["saved_files"]["local_particles"] = len(masses)
    params["saved_files"]["files"] = saved_files
    
    return params
