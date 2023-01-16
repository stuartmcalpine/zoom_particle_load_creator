import os
import subprocess
from string import Template

import particle_load

# Where the template files are located.
_TEMPLATE_DIR = os.path.join(particle_load.__path__[0], "template_files")


def _make_submit_file_swift(swift_dir, params):
    """
    Make slurm submission script using swift template.

    Parameters
    ----------
    swift_dir : string
        Folder where the param files are going
    params : parameter dict
        The paremeters to go into the template
    """

    # Submit file
    template = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/submit"

    with open(template, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    with open("%s/submit" % (swift_dir), "w") as f:
        f.write(result)

    # Resubmit file
    template = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/resubmit"

    with open(template, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    with open("%s/resubmit" % (swift_dir), "w") as f:
        f.write(result)

    # Auto resubmit file
    with open("%s/auto_resubmit" % swift_dir, "w") as f:
        f.write("sbatch resubmit")

    # Change execution privileges (make files executable by group)
    # Assumes the files already exist. If not, it has no effect.
    os.chmod(f"{swift_dir}/submit", 0o744)
    os.chmod(f"{swift_dir}/resubmit", 0o744)
    os.chmod(f"{swift_dir}/auto_resubmit", 0o744)


def _make_param_file_swift(swift_dir, params):
    """
    Make parameter file for swift using template.

    Parameters
    ----------
    swift_dir : string
        Folder where the param files are going
    params : parameter dict
        The parameters to go into the template
    """

    # Starting and finishing scale factors.
    params["starting_a"] = 1.0 / (1 + float(params["starting_z"]))
    params["finishing_a"] = 1.0 / (1 + float(params["finishing_z"]))

    # Replace values.
    # if (
    #    "sibelius" in params["template_set"].lower()
    #    or params["template_set"].lower() == "manticore"
    # ):
    #
    # elif params["template_set"].lower() == "eaglexl":
    #    raise Exception("Fix this one")
    #    # split_mass = gas_particle_mass / 10**10. * 4.
    #    # r = [fname, '%.5f'%h, '%.8f'%starting_a, '%.8f'%finishing_a, '%.8f'%omega0, '%.8f'%omegaL,
    #    #'%.8f'%omegaB, fname, '%.8f'%(eps_dm/h),
    #    #'%.8f'%(eps_baryon/h), '%.8f'%(eps_dm_physical/h),
    #    #'%.8f'%(eps_baryon_physical/h), '%.3f'%(softening_ratio_background),
    #    #'%.8f'%split_mass, ic_dir, fname]
    # else:
    #    raise ValueError("Invalid template set")

    # Some extra params to compute.
    if params["swift_template_set"].lower() == "sibelius_flamingo":
        params["split_mass"] = params["gas_particle_mass"] / 10**10.0 * 4.0

    template = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/params.yml"

    # Replace template values.
    with open(template, "r") as f:
        src = Template(f.read())
        result = src.substitute(params)

    # Write new param file.
    with open("%s/params.yml" % (swift_dir), "w") as f:
        f.write(result)


def make_swift_param_files(params):
    """
    Make submit and parameter file for swift using templates.

    Parameters
    ----------
    params : parameter dict
        The paremeters to go into the template
    """

    # Make main swift folder if it doesn't exist.
    swift_dir = os.path.join(params["save_dir"], "swift")
    if not os.path.exists(swift_dir):
        os.makedirs(swift_dir)
    if not os.path.exists(os.path.join(swift_dir, "out_files/")):
        os.makedirs(os.path.join(swift_dir, "out_files/"))
    if not os.path.exists(os.path.join(swift_dir, "fof/")):
        os.makedirs(os.path.join(swift_dir, "fof/"))
    if not os.path.exists(os.path.join(swift_dir, "snapshots/")):
        os.makedirs(os.path.join(swift_dir, "snapshots/"))

    # Copy over stf times.
    f = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/stf_times_a.txt"
    if os.path.isfile(f):
        subprocess.call(f"cp {f} {swift_dir}", shell=True)

    # Copy over snapshot times.
    f = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/snapshot_times.txt"
    if os.path.isfile(f):
        subprocess.call(f"cp {f} {swift_dir}", shell=True)

    # Copy over select output.
    f = f"{_TEMPLATE_DIR}/swift/{params['swift_template_set']}/select_output.yml"
    if os.path.isfile(f):
        subprocess.call(f"cp {f} {swift_dir}", shell=True)

    # Make submit files.
    _make_submit_file_swift(swift_dir, params)
    print("Saved swift submit file")

    # Make parameter files.
    _make_param_file_swift(swift_dir, params)
    print("Saved swift parameter file")
