import os
from string import Template
import particle_load

# Where the template files are located.
_TEMPLATE_DIR = os.path.join(particle_load.__path__[0], "template_files")


def _make_submit_file_ics(ic_gen_dir, params):
    """
    Make slurm submission script using ic_gen template.

    Parameters
    ----------
    ic_gen_dir : string
        Folder where the param files are going
    params : parameter dict
        The parameters to go into the template
    """

    template = f"{_TEMPLATE_DIR}/ic_gen/{params['ic_gen_template_set']}/submit"

    # Replace template values.
    with open(template, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/submit.sh" % (ic_gen_dir), "w") as f:
        f.write(result)

    # Change execution privileges (make file executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{ic_gen_dir}/submit.sh", 0o744)


def _make_param_file_ics(ic_gen_dir, params):
    """
    Make parameter file for ic_gen using template.

    Parameters
    ----------
    ic_gen_dir : string
        Folder where the param files are going
    params : parameter dict
        The paremeters to go into the template
    """

    template = (
        f"{_TEMPLATE_DIR}/ic_gen/{params['ic_gen_template_set']}/"
        f"params_{params['num_constraint_files']}_con.inp"
    )

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
    with open(template, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.inp" % (ic_gen_dir), "w") as f:
        f.write(result)


def make_ic_param_files(params):
    """
    Make submit and parameter file for ic_gen using templates.

    Parameters
    ----------
    params : parameter dict
        The paremeters to go into the template
    """

    # Make main ic_gen folder if it doesn't exist.
    ic_gen_dir = os.path.join(params["save_dir"], "ic_gen")
    if not os.path.exists(ic_gen_dir):
        os.makedirs(ic_gen_dir)

    # Make output folder for the Ics.
    ic_gen_output_dir = f"{ic_gen_dir}/ICs/"
    if not os.path.exists(ic_gen_output_dir):
        os.makedirs(ic_gen_output_dir)

    # Make submit files.
    _make_submit_file_ics(ic_gen_dir, params)
    print("Saved ic_gen submit file")

    # Make parameter files.
    _make_param_file_ics(ic_gen_dir, params)
    print("Saved ic_gen parameter file")
