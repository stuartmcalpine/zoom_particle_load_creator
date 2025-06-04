import astropy.units as u
from astropy.cosmology import FlatLambdaCDM
from typing import Dict, List, Any, Optional


# Constants for softening rules
SOFTENING_RULES = {
    "flamingo": {
        "comoving_ratio": 1 / 25.0,
        "physical_ratio": 1 / 100.0,  # Transition at z=3.
    },
    "eagle": {
        "comoving_ratio": 1 / 25.0,
        "physical_ratio": 1 / 95.0,  # Transition at z = 2.8
    }
}


def validate_params(params: Dict[str, Any], required_keys: List[str]) -> None:
    """
    Validate that all required keys exist in the params dictionary.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing parameters
    required_keys : List[str]
        List of required key paths (e.g., "cosmology.HubbleParam")
        
    Raises
    ------
    KeyError
        If a required key is missing
    """
    for key_path in required_keys:
        keys = key_path.split('.')
        d = params
        for k in keys:
            if k not in d:
                raise KeyError(f"Missing required parameter: {key_path}")
            d = d[k]


def compute_masses(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    For the given cosmology, compute the total DM mass for the given volume.
    Note: Assuming a flat LCDM cosmology.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing cosmology and simulation parameters.
        Required keys:
        - cosmology.HubbleParam: Hubble parameter (h)
        - cosmology.Omega0: Total matter density parameter
        - cosmology.OmegaBaryon: Baryon density parameter
        - parent.box_size: Box size in h^-1 Mpc
        - parent.n_particles: Number of particles
        
    Returns
    -------
    Dict[str, Any]
        The params dictionary with added mass calculations:
        - M_tot_dm_dmo: Total DM mass in DMO simulation [h^-1 Msol]
        - M_tot_dm_hydro: Total DM mass in hydro simulation [h^-1 Msol]
        - M_tot_gas_hydro: Total gas mass in hydro simulation [h^-1 Msol]
        - M_p_dm_hydro: DM particle mass in hydro simulation [h^-1 Msol]
        - M_p_dm_dmo: DM particle mass in DMO simulation [h^-1 Msol]
        - M_p_gas_hydro: Gas particle mass in hydro simulation [h^-1 Msol]
    """
    # Validate required parameters
    required_keys = [
        "cosmology.HubbleParam", 
        "cosmology.Omega0", 
        "cosmology.OmegaBaryon", 
        "parent.box_size", 
        "parent.n_particles"
    ]
    validate_params(params, required_keys)
    
    # Make a deep copy to avoid modifying the original
    params = params.copy()
    
    # Pull out cosmology from params.
    h = params["cosmology"]["HubbleParam"]
    Om0 = params["cosmology"]["Omega0"]
    Ob0 = params["cosmology"]["OmegaBaryon"]
    box_size = params["parent"]["box_size"] / h  # Mpc
    n_p = params["parent"]["n_particles"]
    
    # Check for invalid values
    if n_p <= 0:
        raise ValueError("Number of particles must be positive")
    if h <= 0:
        raise ValueError("Hubble parameter must be positive")
    if Om0 <= 0:
        raise ValueError("Omega0 must be positive")
    if Ob0 < 0 or Ob0 >= Om0:
        raise ValueError("OmegaBaryon must be non-negative and less than Omega0")
    
    # AstroPy cosmology object.
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=Om0, Ob0=Ob0)
    
    # Critical density in Msol / Mpc**3
    rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc**3)
    
    # Total mass of DM particles (if DMO simulation) in Msol.
    params["cosmology"]["M_tot_dm_dmo"] = Om0 * rho_crit.value * box_size**3
    
    # Total mass of DM and gas (if baryon simulation) in Msol.
    params["cosmology"]["M_tot_dm_hydro"] = (Om0 - Ob0) * rho_crit.value * box_size**3
    params["cosmology"]["M_tot_gas_hydro"] = Ob0 * rho_crit.value * box_size**3
    
    # DM particle mass (if baryon simulation) in Msol.
    params["cosmology"]["M_p_dm_hydro"] = params["cosmology"]["M_tot_dm_hydro"] / n_p
    
    # DM particle mass (if DMO simulation) in Msol.
    params["cosmology"]["M_p_dm_dmo"] = params["cosmology"]["M_tot_dm_dmo"] / n_p
    
    # Gas particle mass (if baryon simulation) in Msol.
    params["cosmology"]["M_p_gas_hydro"] = params["cosmology"]["M_tot_gas_hydro"] / n_p
    
    # Put into inverse h units
    # Create a list of keys to modify before iteration to avoid runtime issues
    mass_keys = [att for att in params["cosmology"].keys() if "M_" in att]
    for att in mass_keys:
        params["cosmology"][att] *= h
        
    return params


def compute_softening(params: Dict[str, Any]) -> Dict[str, Any]:
    """
    Compute softening lengths of the particles.
    
    Parameters
    ----------
    params : Dict[str, Any]
        Dictionary containing simulation parameters.
        Required keys:
        - softening.rule: Softening rule to use ("flamingo", "eagle", or "custom")
        - parent.n_particles: Number of particles
        - parent.box_size: Box size
        
    Returns
    -------
    Dict[str, Any]
        The params dictionary with added softening length calculations:
        - eps_dm: Dark matter softening length (comoving)
        - eps_dm_physical: Dark matter softening length (physical)
        - eps_baryon: Baryon softening length (comoving, for "eagle" or "flamingo" rules)
        - eps_baryon_physical: Baryon softening length (physical, for "eagle" or "flamingo" rules)
        
    Raises
    ------
    NotImplementedError
        If the softening rule is "custom"
    ValueError
        If the softening rule is not recognized
    """
    # Validate required parameters
    required_keys = [
        "softening.rule", 
        "parent.n_particles", 
        "parent.box_size"
    ]
    validate_params(params, required_keys)
    
    # Make a deep copy to avoid modifying the original
    params = params.copy()
    
    # Get softening rule
    rule = params["softening"]["rule"]
    
    # What softening rules to use?
    if rule == "flamingo" or rule == "eagle":
        if rule not in SOFTENING_RULES:
            raise ValueError(f"Softening rule '{rule}' is recognized but not configured in SOFTENING_RULES")
        
        comoving_ratio = SOFTENING_RULES[rule]["comoving_ratio"]
        physical_ratio = SOFTENING_RULES[rule]["physical_ratio"]
    elif rule == "custom":
        raise NotImplementedError(
            "Custom softening rule is not implemented. To implement, add a new entry to "
            "SOFTENING_RULES or modify this function to handle the 'custom' case."
        )
    else:
        raise ValueError(f"Bad softening rule: '{rule}'. Supported values are: {list(SOFTENING_RULES.keys())} and 'custom'")
    
    # Check for valid inputs
    n_p = params["parent"]["n_particles"]
    if n_p <= 0:
        raise ValueError("Number of particles must be positive")
    
    box_size = params["parent"]["box_size"]
    if box_size <= 0:
        raise ValueError("Box size must be positive")
    
    # Cube root of the total number of particles.
    N = n_p ** (1 / 3.0)
    
    # Mean inter-particle separation.
    mean_inter = box_size / N
    
    # Dark matter softening lengths.
    params["softening"]["eps_dm"] = mean_inter * comoving_ratio
    params["softening"]["eps_dm_physical"] = mean_inter * physical_ratio
    
    # Baryon softening lengths.
    if rule in ["eagle", "flamingo"]:
        params["softening"]["eps_baryon"] = params["softening"]["eps_dm"]
        params["softening"]["eps_baryon_physical"] = params["softening"]["eps_dm_physical"]
    
    return params
