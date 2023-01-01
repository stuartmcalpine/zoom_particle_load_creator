import numpy as np


def _compute_ic_cores_from_mem(
    nmaxpart, nmaxdisp, ndim_fft, all_ntot, ncores_node, optimal=False
):
    """
    Compute how many cores we need to run ic_gen on.

    We need to compute the minimum number of cores to fit the IC generation in
    memory, and also the number of cores has to be a multiple of the FFT grid
    size.

    This is based on the fixed parameters (nmaxpart, nmaxdisp) of your ic_gen
    setup, that dictate how much memory gets reserved for the particles and how
    much for the FFT grid. You will find these values in the
    static_memory_grids.inc and static_memory_part.inc files of ic_gen. This
    function also compute what values of nmaxpart and nmaxdisp would be optimal
    for the given run.

    Parameters
    ----------
    nmaxpart : int
        nmaxpart from ic_gen static_memory_part.inc file
        Its the number of particles to assign to each core during ic_gen
    nmaxdisp
        nmaxdisp from ic_gen static_memory_grids.inc file
        Its the number of FFT grid cells to assign to each core during ic_gen
    ndim_fft : int
        FFT grid size (cube root of)
    all_ntot : int
        Total number of particles in particle load
    ncores_node : int
        How many cores per node
    optimal : bool
        Flag for display purposes only

    Returns
    -------
    ncores : int
        Number of cores needed for ic_gen
    """

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

def _compute_optimal_ic_mem(ndim_fft, all_ntot, pl_params):
    """
    This will compute the optimal memory to fit the IC gen on.
    
    Computed based on pl_params.mem_per_core.

    Parameters
    ----------
    ndim_fft : int
        FFT grid size (cube root of)
    all_ntot : int
        Total number of particles in particle load
    pl_params : ParticleLoadParams
        Stores the parameters of the run
    """

    # These are set by ic_gen.
    bytes_per_particle = 66.0
    bytes_per_grid_cell = 20.0

    # Total memory needed for this run.
    total_memory = (bytes_per_particle * all_ntot) + (
        bytes_per_grid_cell * ndim_fft**3.0
    )

    # Compute optimal values of nmaxpart and nmaxdisp
    frac = (bytes_per_particle * all_ntot) / total_memory
    nmaxpart = (frac * pl_params.mem_per_core) / bytes_per_particle

    frac = (bytes_per_grid_cell * ndim_fft**3.0) / total_memory
    nmaxdisp = (frac * pl_params.mem_per_core) / bytes_per_grid_cell

    total_cores = total_memory / pl_params.mem_per_core

    # Print optimal values.
    print("[Optimal] nmaxpart= %i nmaxdisp= %i" % (nmaxpart, nmaxdisp))

    # How many cores needed for optimal setup?
    _compute_ic_cores_from_mem(
        nmaxpart, nmaxdisp, ndim_fft, all_ntot, pl_params.ncores_node, optimal=True
    )


def compute_fft_stats(max_boxsize, all_ntot, pl_params):
    """
    Compute minimum FFT grid size we need for ic_gen.

    For zoom simulations, ic_gen works with nested FFT grids, starting from the
    whole box, halving in size each time a set amount of times. The final grid
    still fully covers the high-res particle region. Each grid has the same FFT
    size. We have to work out, based one the final grid, what the FFT grid size
    should be given the effective resolution if high-res particles were to fill
    that final nested grid entierly.

    For whole volume siulations the FFT grid size is computed in the normal
    way, so that the cell size is at least half the mean inter-particle sep.

    Parameters
    ----------
    max_boxsize : float
        Size of the high-res grid
    all_ntot : int
        Total number of particles in particle load
    pl_params : ParticleLoadParams
        Stores the parameters of the run

    Attributes added to pl_params
    -----------------------------
    high_res_n_eff : int
        Effective resolution in the high-res region
    high_res_L : float
        Size of the high res region
    """

    # Values for FFT over the whole box at target res. True for:
    # - A uniform volume.
    # - A slab simulation.
    # - A non-multigrid ic cubic zoom simulation.
    pl_params.high_res_n_eff = pl_params.n_particles
    pl_params.high_res_L = pl_params.box_size

    # However needs refined for a multigrid cubic zoom simulation.
    if pl_params.is_zoom and pl_params.multigrid_ics:

        # Size of high res region for ic_gen (we use a buffer for saftey).
        pl_params.high_res_L = pl_params.ic_region_buffer_frac * max_boxsize
        assert pl_params.high_res_L < pl_params.box_size, "Zoom buffer region too big"

        if pl_params.high_res_L > pl_params.box_size / 2.0:
            print("--- Cannot use multigrid ICs, zoom region is > boxsize/2.")
        else:
            nlevels = 0
            while pl_params.box_size / (2.0 ** (nlevels + 1)) > pl_params.high_res_L:
                nlevels += 1
            pl_params.high_res_L = pl_params.box_size / (2.0**nlevels)
            pl_params.high_res_n_eff = int(
                pl_params.n_particles
                * (pl_params.high_res_L**3.0 / pl_params.box_size**3)
            )
            print(f"Multigrid ICs with {nlevels} levels")

    print("HRgrid c=%s L_box=%.2f Mpc/h" % (pl_params.coords, pl_params.box_size))
    print(
        "HRgrid L_grid=%.2f Mpc/h n_eff=%.2f**3 (x%.2f=%.2f**3) FFT buff frac= %.2f"
        % (
            pl_params.high_res_L,
            pl_params.high_res_n_eff ** (1 / 3.0),
            pl_params.fft_times_fac,
            pl_params.fft_times_fac * pl_params.high_res_n_eff ** (1 / 3.0),
            pl_params.ic_region_buffer_frac,
        )
    )

    # Minimum FFT grid that fits pl_params.fft_times_fac times (defaut=2) the nyquist frequency.
    ndim_fft = pl_params.ndim_fft_start
    N = (pl_params.high_res_n_eff) ** (1.0 / 3)
    while float(ndim_fft) / float(N) < pl_params.fft_times_fac:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)
    pl_params.ndim_fft = ndim_fft

    # Determine number of cores to use based on memory requirements.
    # Number of cores must also be a factor of ndim_fft.
    print("Using nmaxpart= %i nmaxdisp= %i" % (pl_params.nmaxpart, pl_params.nmaxdisp))
    pl_params.n_cores_ic_gen = _compute_ic_cores_from_mem(
        pl_params.nmaxpart,
        pl_params.nmaxdisp,
        ndim_fft,
        all_ntot,
        pl_params.ncores_node,
        optimal=False,
    )

    # What if we wanted the memory usage to be optimal?
    _compute_optimal_ic_mem(ndim_fft, all_ntot, pl_params)
