import os
import re
import subprocess
from string import Template

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
        pl_params.pl_basename = (
            f"./ic_gen_submit_files/{pl_params.f_name}/particle_load/fbinary/PL"
        )
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
        "f_name",
        "n_species",
        "ic_dir",
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
        "swift_ic_dir_loc",
        "softening_ratio_background",
        "template_set",
        "gas_particle_mass",
        "swift_dir",
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

    return param_dict


def make_submit_file_ics(params):
    """
    Make slurm submission script using ic_gen template.

    Parameters
    ----------
    params : parameter dict
        The paremeters to go into the template

    Returns
    -------
    None
        Creates the submit files
    """

    # Make folder if it doesn't exist.
    ic_gen_dir = "%s/%s/" % (params["ic_dir"], params["f_name"])
    if not os.path.exists(ic_gen_dir):
        os.makedirs(ic_gen_dir)

    # Replace template values.
    with open("./templates/ic_gen/submit", "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/submit.sh" % (ic_gen_dir), "w") as f:
        f.write(result)

    # Change execution privileges (make file executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{ic_gen_dir}/submit.sh", 0o744)


def make_param_file_ics(params):
    """
    Make parameter file for ic_gen using template.

    Parameters
    ----------
    params : parameter dict
        The paremeters to go into the template

    Returns
    -------
    None
        Creates the parameter file
    """

    # Make folder if it doesn't exist.
    ic_gen_dir = "%s/%s/" % (params["ic_dir"], params["f_name"])
    if not os.path.exists(ic_gen_dir):
        os.makedirs(ic_gen_dir)

    # Make output folder for the Ics.
    ic_gen_output_dir = "%s/%s/ICs/" % (
        params["ic_dir"],
        params["f_name"],
    )
    if not os.path.exists(ic_gen_output_dir):
        os.makedirs(ic_gen_output_dir)

    # Minimum FFT grid that fits 2x the nyquist frequency.
    ndim_fft = params["ndim_fft_start"]
    N = (
        (params["high_res_n_eff"]) ** (1.0 / 3)
        if params["is_zoom"]
        else (params["n_particles"]) ** (1 / 3.0)
    )
    while float(ndim_fft) / float(N) < params["fft_times_fac"]:
        ndim_fft *= 2
    print("Using ndim_fft = %d" % ndim_fft)
    params["ndim_fft"] = ndim_fft

    # Is this a zoom simulation (zoom can't use 2LPT)?
    if params["is_zoom"]:
        if params["is_slab"]:
            params["two_lpt"] = 1
            params["multigrid"] = 0
        else:
            params["two_lpt"] = 0 if params["multigrid_ics"] else 1
            params["multigrid"] = 1 if params["multigrid_ics"] else 0
    else:
        params["high_res_L"] = 0.0
        params["high_res_n_eff"] = 0
        params["two_lpt"] = 1
        params["multigrid"] = 0

    # Use peano hilbert indexing?
    params["use_ph"] = 2 if params["use_ph_ids"] else 1

    # Cube of neff
    params["high_res_n_eff_cube"] = round(params["high_res_n_eff"] ** (1.0 / 3))

    # Replace template values.
    with open(
            f"./templates/ic_gen/params_{params['num_constraint_files']}_con.inp", "r"
    ) as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.inp" % (ic_gen_dir), "w") as f:
        f.write(result)

def make_param_file_swift(params):
    """
    Make parameter file for swift using template.

    Parameters
    ----------
    params : parameter dict
        The parameters to go into the template

    Returns
    -------
    None
        Creates the parameter file
    """

    # Make data dir.
    data_dir = params["swift_dir"] + "%s/" % params["f_name"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
    if not os.path.exists(data_dir + "out_files/"):
        os.makedirs(data_dir + "out_files/")
    if not os.path.exists(data_dir + "fof/"):
        os.makedirs(data_dir + "fof/")
    if not os.path.exists(data_dir + "snapshots/"):
        os.makedirs(data_dir + "snapshots/")

    # Starting and finishing scale factors.
    params["starting_a"] = 1.0 / (1 + float(params["starting_z"]))
    params["finishing_a"] = 1.0 / (1 + float(params["finishing_z"]))

    # Replace values.
    if (
        "sibelius" in params["template_set"].lower()
        or params["template_set"].lower() == "manticore"
    ):
        # Copy over stf times.
        f = f"./templates/swift/{params['template_set']}/stf_times_a.txt"
        if os.path.isfile(f):
            subprocess.call(f"cp {f} {data_dir}", shell=True)

        # Copy over snapshot times.
        f = f"./templates/swift/{params['template_set']}/snapshot_times.txt"
        if os.path.isfile(f):
            subprocess.call(f"cp {f} {data_dir}", shell=True)

        # Copy over select output.
        f = f"./templates/swift/{params['template_set']}/select_output.yml"
        if os.path.isfile(f):
            subprocess.call(f"cp {f} {data_dir}", shell=True)
    
    elif params["template_set"].lower() == "eaglexl":
        raise Exception("Fix this one")
        # split_mass = gas_particle_mass / 10**10. * 4.
        # r = [fname, '%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
        #'%.8f'%omegaB, fname, '%.8f'%(eps_dm/h),
        #'%.8f'%(eps_baryon/h), '%.8f'%(eps_dm_physical/h),
        #'%.8f'%(eps_baryon_physical/h), '%.3f'%(softening_ratio_background),
        #'%.8f'%split_mass, ic_dir, fname]
    else:
        raise ValueError("Invalid template set")

    # Some extra params to compute.
    if params["template_set"].lower() == "sibelius_flamingo":
        params["split_mass"] = params["gas_particle_mass"] / 10**10.0 * 4.0

    t_file = "./templates/swift/%s/params.yml" % params["template_set"]

    # Replace template values.
    with open(t_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.yml" % (data_dir), "w") as f:
        f.write(result)


def make_submit_file_swift(params):
    """
    Make slurm submission script using swift template.

    Parameters
    ----------
    params : parameter dict
        The paremeters to go into the template

    Returns
    -------
    None
        Creates the submit files
    """

    data_dir = params["swift_dir"] + "%s/" % params["f_name"]
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    s_file = "./templates/swift/%s/submit" % params["template_set"].lower()
    rs_file = "./templates/swift/%s/resubmit" % params["template_set"].lower()

    # Replace template values.
    with open(s_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/submit" % (data_dir), "w") as f:
        f.write(result)

    # Replace template values.
    with open(rs_file, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/resubmit" % (data_dir), "w") as f:
        f.write(result)

    with open("%s/auto_resubmit" % data_dir, "w") as f:
        f.write("sbatch resubmit")

    # Change execution privileges (make files executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{data_dir}/submit", 0o744)
    os.chmod(f"{data_dir}/resubmit", 0o744)
    os.chmod(f"{data_dir}/auto_resubmit", 0o744)
