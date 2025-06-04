import os
import numpy as np
import yaml
import h5py
import toml

import particle_load.mympi as mympi
from .cosmology import compute_masses, compute_softening

# Default parameters
_DEFAULTS = {
    "zoom": {
        "mask": None,
        "enable": False,
        "min_nq": 20,
        "_max_nq": 1000,
        "grid_also_glass": True,
        "glass_buffer_cells": 3,
        "min_num_per_cell": 8,
        "skin_reduce_factor": 1 / 8.0,
        "to_plot": False,
        "nq_mass_reduce_factor": 1 / 2.0,
        "ic_region_buffer_frac": 1.0,
    },
    "swift": {
        "n_nodes": 1,
        "n_hours": 10,
        "exec": ".",
        "softening_ratio_background": 0.02,
        "swift_template_set": "dmo",
        "starting_z": 127.0,
        "finishing_z": 0.0,
    },
    "ic_gen": {
        "nmaxpart": 36045928,
        "nmaxdisp": 791048437,
        "mem_per_core": 18e9,
        "max_particles_per_ic_file": 400**3,
        "use_ph_ids": True,
        "nbit": 21,
        "fft_times_fac": 2.0,
        "ncores_node": 28,
        "import_file": 0,
        "pl_rep_factor": 1,
        "num_hours": 10,
        "exec": None,
        "multigrid_ics": True,
        "starting_z": 127.0,
        "num_constraint_files": 0,
    },
    "system": {"verbose": True},
}


def _compute_numbers(params):
    """
    Calculate derived values based on parameter file.
    
    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters
    """
    # Calculate how many glass cells would fill the whole simulation volume
    # (Cube root of total)
    params["parent"]["N_glass_L"] = int(
        np.rint(
            (params["parent"]["n_particles"] / params["glass_file"]["N"]) ** (1 / 3.0)
        )
    )
    
    # Verify calculation is correct
    assert (
        params["parent"]["N_glass_L"] ** 3 * params["glass_file"]["N"]
        == params["parent"]["n_particles"]
    )

    # Size of a glass cell in Mpc/h
    params["glass_file"]["L_mpch"] = np.true_divide(
        params["parent"]["box_size"], params["parent"]["N_glass_L"]
    )

    # Extract ndim_fft_start from the descriptor
    descriptor_parts = params["ic_gen"]["panphasian_descriptor"].split(",")
    if len(descriptor_parts) < 6 or "S" not in descriptor_parts[5]:
        raise ValueError("Invalid panphasian descriptor format")
    
    tmp = descriptor_parts[5]
    params["ic_gen"]["ndim_fft_start"] = int(tmp.split("S")[1])

    # Set number of particle species
    params["ic_gen"]["n_species"] = 2 if params["zoom"]["enable"] else 1


def _check_params(params):
    """
    Verify parameter validity.
    
    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters
        
    Raises
    ------
    ValueError
        If parameters are invalid
    FileNotFoundError
        If required files don't exist
    """
    # Verify particle numbers are compatible with glass file
    if (
        np.true_divide(params["parent"]["n_particles"], params["glass_file"]["N"]) % 1
        > 1e-6
    ):
        raise ValueError("Number of particles must divide evenly into glass_num")

    # Check softening rule is valid
    allowed_rules = ["custom", "eagle", "flamingo"]
    if params["softening"]["rule"] not in allowed_rules:
        raise ValueError(f"Softening rule '{params['softening']['rule']}' is not allowed. Must be one of {allowed_rules}")

    # Verify zoom mask exists if zoom is enabled
    if params["zoom"]["enable"]:
        if params["zoom"]["mask"] is None:
            raise ValueError("Must provide zoom mask when zoom is enabled")

        if not os.path.isfile(params["zoom"]["mask"]):
            raise FileNotFoundError(f"Zoom mask file not found: {params['zoom']['mask']}")

    # Check constraint descriptors
    assert params["ic_gen"]["num_constraint_files"] <= 2
    if params["ic_gen"]["num_constraint_files"] > 0:
        for i in range(params["ic_gen"]["num_constraint_files"]):
            assert f"constraint_phase_descriptor_{i+1}" in params["ic_gen"]
            assert f"constraint_phase_descriptor_path_{i+1}" in params["ic_gen"]
            assert f"constraint_phase_descriptor_levels_{i+1}" in params["ic_gen"]

def _print_params(params):
    """
    Print parameters in a formatted way.
    
    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters
    """
    for category in params:
        print(f"{category}")
        print("-" * len(category))
        
        for attr in params[category]:
            # Skip printing large coordinate arrays
            if attr == "mask_coordinates":
                continue
                
            # Format large numbers differently
            if isinstance(params[category][attr], (float, np.float64, np.float32)) and params[category][attr] > 1e4:
                print(f"\033[96m{attr}\033[0m: \033[92m{params[category][attr]:.4g}\033[0m")
                continue

            print(f"\033[96m{attr}\033[0m: \033[92m{params[category][attr]}\033[0m")


def _load_zoom_mask(params):
    """
    Load the zoom mask from an HDF5 file.
    
    Parameters
    ----------
    params : dict
        Dictionary containing simulation parameters
    """
    with h5py.File(params["zoom"]["mask"], "r") as f:
        # Load attributes from the Coordinates group
        for attr in ["bounding_length", "high_res_volume", "grid_cell_width", "geo_centre"]:
            params["zoom"][attr] = f["Coordinates"].attrs.get(attr)

        # Load the coordinates array
        params["zoom"]["mask_coordinates"] = np.array(f["Coordinates"][...], dtype="f8")


def read_param_file(args):
    """
    Read parameters from a specified TOML file.

    See template file `param_files/template.yml` for a full listing and
    description of parameters.

    In the MPI version, the file is only read on one rank and the dict
    is then broadcast to all other ranks.

    Parameters
    ----------
    args : argparse object
        Command-line arguments, must contain a 'param_file' attribute

    Returns
    -------
    params : dict
        Dictionary containing all simulation parameters
    """
    # Load parameters from TOML file
    params = toml.load(args.param_file)

    # Fill in defaults for any missing parameters
    for category in _DEFAULTS:
        if category in params:
            for attr in _DEFAULTS[category]:
                if attr not in params[category]:
                    params[category][attr] = _DEFAULTS[category][attr]
        else:
            params[category] = _DEFAULTS[category].copy()

    # Check for required parameters
    required_params = {
        "parent": ["box_size", "n_particles"],
        "output": ["path"],
        "cosmology": [
            "Omega0",
            "OmegaCDM",
            "OmegaLambda",
            "OmegaBaryon",
            "HubbleParam",
            "Sigma8",
        ],
        "glass_file": ["N"],
        "ic_gen": ["panphasian_descriptor", "linear_ps"],
        "softening": ["rule"],
    }

    for category in required_params:
        if category not in params:
            raise ValueError(f"Missing category: {category}")
            
        for attr in required_params[category]:
            if attr not in params[category]:
                raise ValueError(f"Missing required parameter: {category}:{attr}")

    # Store command-line arguments in params dict
    params["cmd_args"] = vars(args)

    # Validate parameters
    _check_params(params)

    # Calculate derived parameters
    _compute_numbers(params)
    compute_masses(params)
    compute_softening(params)

    # Load zoom mask if zoom is enabled
    if params["zoom"]["enable"]:
        _load_zoom_mask(params)
        params["parent"]["coords"] = params["zoom"]["geo_centre"]
    else:
        params["parent"]["coords"] = None

    # Print parameter summary
    _print_params(params)

    return params


class ParticleLoadParams:
    """
    Legacy class for parameter handling.
    
    Note: This appears to be partially replaced by the function-based approach above.
    Kept for backward compatibility.
    """
    
    def __init__(self, args):
        """
        Initialize parameter object.
        
        Parameters
        ----------
        args : object
            Should have attributes used to set parameters
        """
        # Load zoom region mask file if specified
        if hasattr(self, 'mask_file') and self.mask_file is not None:
            self._load_high_res_region_mask()

    def _load_high_res_region_mask(self):
        """Load the mask HDF5 file."""
        # This appears to use a ZoomRegionMask class that's not defined in the provided files
        self.high_res_region_mask = ZoomRegionMask(self.mask_file)

        # Set coordinates using the mask's geometric center
        self.coords = self.high_res_region_mask.geo_centre
        print(f"Set coordinates of high res region using mask: {self.coords}")

    def _sanity_checks(self):
        """Perform basic sanity checks on parameters."""
        # Ensure coords is a numpy array
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords)

        # Check IC generator paths if making IC files
        if self.make_ic_gen_param_files:
            assert hasattr(self, "ic_gen_exec")

        # Check SWIFT paths if making SWIFT files
        if self.make_swift_param_files:
            assert hasattr(self, "swift_template_set")
            assert hasattr(self, "swift_exec")

        # Check non-zoom simulation parameters
        if not self.is_zoom:
            assert self.multigrid_ics == 0, "multigrid_ics must be 0 for non-zoom simulations"
            assert hasattr(self, "glass_file_loc")

        # Verify mask file is only used with zoom simulations
        if self.mask_file is not None:
            assert self.is_zoom, "Mask file can only be used with zoom simulations"

    def _populate_defaults(self):
        """Set default values for parameters if not already specified."""
        self._add_default_value("coords", np.array([0.0, 0.0, 0.0]))
        self._add_default_value("ic_region_buffer_frac", 1.0)
        self._add_default_value("is_slab", False)
