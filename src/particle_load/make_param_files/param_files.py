import os
import re

import numpy as np


def build_param_dict(params, high_res_region):
    """
    Some extra quantities are also computed, params dict updated in place.

    Parameters
    ----------
    params : dict
        Stores the parameters of the run
    high_res_region : HighResolutionRegion object
        Stores the information about the high res zoom region-
    """

    # Compute mass cut offs between highest mass high-res-grid particles and
    # lowest mass skin particles.
    sp1_sp2_cut = 0.0
    if params["zoom"]["enable"]:
        params["ic_gen"]["sp1_sp2_cut"] = np.log10(high_res_region.cell_info["glass_mass"]) + 0.01
        print(f"Sp1->Sp2 cut: {params['ic_gen']['sp1_sp2_cut']}")

        # For zooms, where is the particle load?
        params["ic_gen"]["pl_basename"] = "../particle_load/fbinary/PL"

    # For uniform volumes, where is the glass file?
    else:
        raise NotImplementedError
        #pl_params.import_file = -1
        #pl_params.pl_basename = pl_params.glass_file_loc
        #tmp_glass_no = int(pl_params.pl_basename.split("_")[-1])
        #assert tmp_glass_no == pl_params.glass_num, "Glass num and file not match"
        #pl_params.pl_rep_factor = int(
        #    np.rint((pl_params.n_particles / pl_params.glass_num) ** (1 / 3.0))
        #)
        #assert (
        #    pl_params.pl_rep_factor**3 * pl_params.glass_num == pl_params.n_particles
        #), "Error rep_factor"
        #print(
        #    f"Uniform volume {pl_params.glass_num**(1/3.):.0f}**3 replicated",
        #    f"{pl_params.pl_rep_factor}**3 times",
        #)

    # Copy some information
    params["ic_gen"]["high_res_L"] = params["zoom"]["high_res_L"]
    params["ic_gen"]["high_res_n_eff"] = params["zoom"]["high_res_n_eff"]
    params["ic_gen"]["box_size"] = params["parent"]["box_size"]
    params["ic_gen"]["n_particles"] = params["parent"]["n_particles"]
    for att in ["ic_gen", "swift"]:
        for att2 in ["Omega0", "OmegaLambda", "HubbleParam", "Sigma8", "OmegaCDM", "OmegaBaryon"]:
            params[att][att2] = params["cosmology"][att2]
    params["ic_gen"]["coords_x"] = params["parent"]["coords"][0]
    params["ic_gen"]["coords_y"] = params["parent"]["coords"][1]
    params["ic_gen"]["coords_z"] = params["parent"]["coords"][2]

    # Number of particles types.
    if params["zoom"]["enable"]:
        params["n_species"] = 2
    else:
        params["n_species"] = 1

    # Softening lenghts
    params["swift"]["eps_dm"] = "%.8f" % (params["softening"]["eps_dm"] / params["cosmology"]["HubbleParam"])
    params["swift"]["eps_baryon"] = "%.8f" % (params["softening"]["eps_baryon"] / params["cosmology"]["HubbleParam"])
    params["swift"]["eps_dm_physical"] = "%.8f" % (params["softening"]["eps_dm_physical"] / params["cosmology"]["HubbleParam"])
    params["swift"]["eps_baryon_physical"] = "%.8f" % (params["softening"]["eps_baryon_physical"] / params["cosmology"]["HubbleParam"])

