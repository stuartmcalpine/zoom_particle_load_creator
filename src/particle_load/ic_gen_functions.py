import numpy as np
from typing import Dict, Any, Union, Optional
from copy import deepcopy

# Constants for memory calculations
BYTES_PER_PARTICLE = 66.0
BYTES_PER_GRID_CELL = 20.0


def validate_params(params: Dict[str, Any], required_keys: Dict[str, list]) -> None:
    """
    Validate that all required keys exist in the params dictionary.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing parameters
    required_keys : Dict[str, list]
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


def _compute_ic_cores_from_mem(
    nmaxpart: int, 
    nmaxdisp: int, 
    ndim_fft: int, 
    all_ntot: int, 
    ncores_node: int, 
    optimal: bool = False
) -> int:
    """
    Compute how many cores we need to run ic_gen on.

    We need to compute the minimum number of cores to fit the IC generation in
    memory, and also the number of cores has to be a multiple of the FFT grid
    size.

    This is based on the fixed parameters (nmaxpart, nmaxdisp) of your ic_gen
    setup, that dictate how much memory gets reserved for the particles and how
    much for the FFT grid. You will find these values in the
    static_memory_grids.inc and static_memory_part.inc files of ic_gen. This
    function also computes what values of nmaxpart and nmaxdisp would be optimal
    for the given run (but obviously you would then have to change and
    recompile ic_gen if you want to use them).

    Parameters
    ----------
    nmaxpart : int
        nmaxpart from ic_gen static_memory_part.inc file
        Its the number of particles to assign to each core during ic_gen
    nmaxdisp : int
        nmaxdisp from ic_gen static_memory_grids.inc file
        Its the number of FFT grid cells to assign to each core during ic_gen
    ndim_fft : int
        FFT grid size (cube root of)
    all_ntot : int
        Total number of particles in particle load
    ncores_node : int
        How many cores per node
    optimal : bool, optional
        Flag for display purposes only, default is False

    Returns
    -------
    ncores : int
        Number of cores needed for ic_gen
    """
    # Input validation
    if nmaxpart <= 0:
        raise ValueError("nmaxpart must be positive")
    if nmaxdisp <= 0:
        raise ValueError("nmaxdisp must be positive")
    if ndim_fft <= 0:
        raise ValueError("ndim_fft must be positive")
    if all_ntot <= 0:
        raise ValueError("all_ntot must be positive")
    if ncores_node <= 0:
        raise ValueError("ncores_node must be positive")

    # Number of cores needed for FFT grid.
    ncores_ndisp = np.ceil(
        float((ndim_fft * ndim_fft * 2 * (ndim_fft / 2 + 1))) / nmaxdisp
    )

    # Number of cores needed for particles.
    ncores_npart = np.ceil(float(all_ntot) / nmaxpart)

    # Minimum number of cores needed.
    ncores = max(ncores_ndisp, ncores_npart)

    # Number of cores must fit into FFT grid size.
    while (ndim_fft % ncores) != 0:
        ncores += 1

    # If we're using one node, try to use as many of the cores as possible
    if ncores < ncores_node:
        ncores = ncores_node
        while (ndim_fft % ncores) != 0:
            ncores -= 1

    # Print result.
    this_str = "[Optimal] " if optimal else ""
    print(
        "%sUsing %i cores for IC gen (min %i for FFT and min %i for particles)"
        % (this_str, ncores, ncores_ndisp, ncores_npart)
    )

    return ncores


def _compute_optimal_ic_mem(
    ndim_fft: int, 
    all_ntot: int, 
    params: Dict[str, Any]
) -> None:
    """
    This will compute the optimal memory to fit the IC gen on.

    Computed based on mem_per_core.

    Parameters
    ----------
    ndim_fft : int
        FFT grid size (cube root of)
    all_ntot : int
        Total number of particles in particle load
    params : Dict[str, Any]
        Dictionary containing simulation parameters
        Required keys:
        - ic_gen.mem_per_core: Memory per core in bytes
        - ic_gen.ncores_node: Number of cores per node
        
    Returns
    -------
    None
        This function modifies the params dictionary in place and prints results
    """
    # Input validation
    required_keys = {"ic_gen": ["mem_per_core", "ncores_node"]}
    validate_params(params, required_keys)
    
    if ndim_fft <= 0:
        raise ValueError("ndim_fft must be positive")
    if all_ntot <= 0:
        raise ValueError("all_ntot must be positive")
    if params["ic_gen"]["mem_per_core"] <= 0:
        raise ValueError("mem_per_core must be positive")
    if params["ic_gen"]["ncores_node"] <= 0:
        raise ValueError("ncores_node must be positive")

    # Total memory needed for this run.
    total_memory = (BYTES_PER_PARTICLE * all_ntot) + (
        BYTES_PER_GRID_CELL * ndim_fft**3.0
    )

    # Compute optimal values of nmaxpart and nmaxdisp
    frac_particles = (BYTES_PER_PARTICLE * all_ntot) / total_memory
    nmaxpart_float = (frac_particles * params["ic_gen"]["mem_per_core"]) / BYTES_PER_PARTICLE
    nmaxpart = int(nmaxpart_float)  # Convert to integer for use in _compute_ic_cores_from_mem

    frac_grid = 1 - frac_particles
    nmaxdisp_float = (frac_grid * params["ic_gen"]["mem_per_core"]) / BYTES_PER_GRID_CELL
    nmaxdisp = int(nmaxdisp_float)  # Convert to integer for use in _compute_ic_cores_from_mem

    # Calculate total cores needed (for informational purposes)
    total_cores_needed = total_memory / params["ic_gen"]["mem_per_core"]

    # Print optimal values.
    print(
        f"mem_per_core={params['ic_gen']['mem_per_core']} ncores_node={params['ic_gen']['ncores_node']}"
    )
    print(f"[Optimal] nmaxpart= {nmaxpart} nmaxdisp= {nmaxdisp}")
    print(f"[Optimal] Total cores needed (memory-based): {total_cores_needed:.2f}")

    # How many cores needed for optimal setup?
    _compute_ic_cores_from_mem(
        nmaxpart,
        nmaxdisp,
        ndim_fft,
        all_ntot,
        params["ic_gen"]["ncores_node"],
        optimal=True,
    )


def compute_fft_stats(
    max_boxsize: float, 
    all_ntot: int, 
    params: Dict[str, Any]
):
    """
    Compute minimum FFT grid size we need for ic_gen.

    For zoom simulations, ic_gen works with nested FFT grids, starting from the
    whole box, halving in size each time a set amount of times. The final grid
    still fully covers the high-res particle region. Each grid has the same FFT
    size. We have to work out, based on the final grid, what the FFT grid size
    should be given the effective resolution if high-res particles were to fill
    that final nested grid entirely.

    For whole volume simulations the FFT grid size is computed in the normal
    way, so that the cell size is at least half the mean inter-particle separation.

    Parameters
    ----------
    max_boxsize : float
        Size of the high-res grid
    all_ntot : int
        Total number of particles in particle load
    params : Dict[str, Any]
        Dictionary containing simulation parameters
        Required keys:
        - parent.n_particles: Number of particles in parent simulation
        - parent.box_size: Box size of parent simulation
        - parent.coords: Coordinates of parent simulation
        - zoom.enable: Whether zoom simulation is enabled
        - zoom.ic_region_buffer_frac: Buffer fraction for IC region
        - ic_gen.multigrid_ics: Whether multigrid ICs are enabled
        - ic_gen.ndim_fft_start: Starting FFT dimension
        - ic_gen.fft_times_fac: FFT times factor
        - ic_gen.nmaxpart: Maximum particles per core
        - ic_gen.nmaxdisp: Maximum FFT cells per core
        - ic_gen.ncores_node: Number of cores per node
    """
    # Input validation
    required_keys = {
        "parent": ["n_particles", "box_size", "coords"],
        "zoom": ["enable", "ic_region_buffer_frac"],
        "ic_gen": [
            "multigrid_ics", "ndim_fft_start", "fft_times_fac",
            "nmaxpart", "nmaxdisp", "ncores_node"
        ]
    }
    validate_params(params, required_keys)
    
    if max_boxsize <= 0:
        raise ValueError("max_boxsize must be positive")
    if all_ntot <= 0:
        raise ValueError("all_ntot must be positive")
    if params["parent"]["n_particles"] <= 0:
        raise ValueError("n_particles must be positive")
    if params["parent"]["box_size"] <= 0:
        raise ValueError("box_size must be positive")
    
    # Values for FFT over the whole box at target res. True for:
    # - A uniform volume.
    # - A slab simulation.
    # - A non-multigrid ic cubic zoom simulation.
    params["zoom"]["high_res_n_eff"] = params["parent"]["n_particles"]
    params["zoom"]["high_res_L"] = params["parent"]["box_size"]

    # However needs refined for a multigrid cubic zoom simulation.
    if params["zoom"]["enable"] and params["ic_gen"]["multigrid_ics"]:

        # Size of high res region for ic_gen (we use a buffer for safety).
        params["zoom"]["high_res_L"] = (
            params["zoom"]["ic_region_buffer_frac"] * max_boxsize
        )
        
        # Check if high-res region is smaller than parent box
        if params["zoom"]["high_res_L"] >= params["parent"]["box_size"]:
            raise ValueError("Zoom buffer region too big")

        if params["zoom"]["high_res_L"] > params["parent"]["box_size"] / 2.0:
            print("--- Cannot use multigrid ICs, zoom region is > boxsize/2.")
            params["ic_gen"]["multigrid_ics"] = False
        else:
            nlevels = 0
            while (
                params["parent"]["box_size"] / (2.0 ** (nlevels + 1))
                > params["zoom"]["high_res_L"]
            ):
                nlevels += 1
            params["zoom"]["high_res_L"] = params["parent"]["box_size"] / (
                2.0**nlevels
            )
            params["zoom"]["high_res_n_eff"] = int(
                params["parent"]["n_particles"]
                * (
                    params["zoom"]["high_res_L"] ** 3.0
                    / params["parent"]["box_size"] ** 3
                )
            )
            print(f"Multigrid ICs with {nlevels} levels")

    print(
        "HRgrid c=%s L_box=%.2f Mpc/h"
        % (params["parent"]["coords"], params["parent"]["box_size"])
    )
    print(
        "HRgrid L_grid=%.2f Mpc/h n_eff=%.2f**3 (x%.2f=%.2f**3) FFT buff frac= %.2f"
        % (
            params["zoom"]["high_res_L"],
            params["zoom"]["high_res_n_eff"] ** (1 / 3.0),
            params["ic_gen"]["fft_times_fac"],
            params["ic_gen"]["fft_times_fac"]
            * params["zoom"]["high_res_n_eff"] ** (1 / 3.0),
            params["zoom"]["ic_region_buffer_frac"],
        )
    )

    # Minimum FFT grid that fits fft_times_fac times (default=2) the nyquist frequency.
    ndim_fft = params["ic_gen"]["ndim_fft_start"]
    particles_per_dimension = (params["zoom"]["high_res_n_eff"]) ** (1.0 / 3)
    
    while float(ndim_fft) / float(particles_per_dimension) < params["ic_gen"]["fft_times_fac"]:
        ndim_fft *= 2
    
    print("Using ndim_fft = %d" % ndim_fft)
    params["ic_gen"]["ndim_fft"] = ndim_fft

    # Determine number of cores to use based on memory requirements.
    # Number of cores must also be a factor of ndim_fft.
    print(
        "Using nmaxpart= %i nmaxdisp= %i"
        % (params["ic_gen"]["nmaxpart"], params["ic_gen"]["nmaxdisp"])
    )
    params["ic_gen"]["n_cores"] = int(_compute_ic_cores_from_mem(
        params["ic_gen"]["nmaxpart"],
        params["ic_gen"]["nmaxdisp"],
        ndim_fft,
        all_ntot,
        params["ic_gen"]["ncores_node"],
        optimal=False,
    ))

    # What if we wanted the memory usage to be optimal?
    _compute_optimal_ic_mem(ndim_fft, all_ntot, params)
