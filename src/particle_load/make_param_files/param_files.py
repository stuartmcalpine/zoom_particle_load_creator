import os
import re

import numpy as np


def build_param_dict(pl_params, high_res_region):
    """
    Put together a dict of parameters to feed into the template files.

    Essentially pulls out the parameters from pl_params and puts them in a dict.

    Some extra quantities are also computed.

    Parameters
    ----------
    pl_params : ParticleLoadParams object
        Stores the parameters of the run
    high_res_region : HighResolutionRegion object
        Stores the information about the high res zoom region-

    Returns
    -------
    param_dict : dict
        Dictionary of params
    """

    # Compute mass cut offs between particle types.
    sp1_sp2_cut = 0.0
    if pl_params.is_zoom:
        sp1_sp2_cut = np.log10(high_res_region.cell_info["glass_mass"]) + 0.01

    # Glass files for non zooms.
    if pl_params.is_zoom:
        pl_params.pl_basename = "../particle_load/fbinary/PL"
    else:
        pl_params.import_file = -1
        pl_params.pl_basename = pl_params.glass_file_loc
        pl_params.pl_rep_factor = int(
            np.rint((pl_params.n_particles / pl_params.glass_num) ** (1 / 3.0))
        )
        assert (
            pl_rep_factor**3 * pl_params.glass_num == pl_params.n_particles
        ), "Error rep_factor"

    # Build parameter list.
    param_dict = dict(
        sp1_sp2_cut="%.3f" % sp1_sp2_cut,
        coords_x="%.8f" % pl_params.coords[0],
        coords_y="%.8f" % pl_params.coords[1],
        coords_z="%.8f" % pl_params.coords[2],
        eps_dm="%.8f" % (pl_params.eps_dm / pl_params.HubbleParam),
        eps_baryon="%.8f" % (pl_params.eps_baryon / pl_params.HubbleParam),
        eps_dm_physical="%.8f" % (pl_params.eps_dm_physical / pl_params.HubbleParam),
        eps_baryon_physical="%.8f"
        % (pl_params.eps_baryon_physical / pl_params.HubbleParam),
    )

    for att in [
        "import_file",
        "pl_basename",
        "pl_rep_factor",
        "save_dir",
        "n_particles",
        "high_res_n_eff",
        "panphasian_descriptor",
        "ndim_fft_start",
        "is_slab",
        "use_ph_ids",
        "multigrid_ics",
        "linear_ps",
        "nbit",
        "fft_times_fac",
        "ndim_fft",
        "swift_ic_dir_loc",
        "softening_ratio_background",
        "ic_gen_template_set",
        "ic_gen_exec",
        "swift_template_set",
        "swift_exec",
        "gas_particle_mass",
        "num_constraint_files",
        "num_hours_ic_gen",
        "swift_exec_location",
        "num_hours_swift",
    ]:
        param_dict[att] = getattr(pl_params, att)

    for att in [
        "high_res_L",
        "Omega0",
        "OmegaCDM",
        "OmegaLambda",
        "OmegaBaryon",
        "HubbleParam",
        "Sigma8",
        "starting_z",
        "finishing_z",
        "box_size",
    ]:
        param_dict[att] = f"{getattr(pl_params, att):.8f}"

    for att in ["n_nodes_swift", "n_cores_ic_gen", "is_zoom"]:
        param_dict[att] = f"{int(getattr(pl_params, att))}"

    for i in range(pl_params.num_constraint_files):
        param_dict[f"constraint_phase_descriptor_{i+1}"] = getattr(
            pl_params, f"constraint_phase_descriptor_{i+1}"
        )
        param_dict[f"constraint_phase_descriptor_{i+1}_path"] = getattr(
            pl_params, f"constraint_phase_descriptor_{i+1}_path"
        )
        param_dict[f"constraint_phase_descriptor_{i+1}_levels"] = getattr(
            pl_params, f"constraint_phase_descriptor_{i+1}_levels"
        )

    if pl_params.is_zoom:
        param_dict['n_species'] = 2
    else:
        param_dict['n_species'] = 1

    return param_dict
