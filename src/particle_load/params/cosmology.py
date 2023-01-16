import astropy.units as u
from astropy.cosmology import FlatLambdaCDM


def compute_masses(pl_params):
    """
    For the given cosmology, compute the total DM mass for the given volume.

    Note: Assuming a flat LCDM cosmology.

    Parameters
    ----------
    pl_params : PLParams object

    Adds to pl_params object
    ------------------------
    total_box_mass : float
        The total mass of particles in the simulation in Msol
    gas_particle_mass : float
        The gas particle mass in Msol
    """

    # Pull out cosmology from params.
    h = pl_params.HubbleParam
    Om0 = pl_params.Omega0
    Ob0 = pl_params.OmegaBaryon
    box_size = pl_params.box_size

    # AstroPy cosmology object.
    cosmo = FlatLambdaCDM(H0=h * 100.0, Om0=Om0, Ob0=Ob0)

    # Critical density in Msol / Mpc**3
    rho_crit = cosmo.critical_density0.to(u.solMass / u.Mpc**3)

    # Total mass of DM particles (if DMO simulation) in Msol.
    M_tot_dm_dmo = Om0 * rho_crit.value * (box_size / h) ** 3

    # Total mass of DM and gas (if baryon simulation) in Msol.
    M_tot_dm = (Om0 - Ob0) * rho_crit.value * (box_size / h) ** 3
    M_tot_gas = Ob0 * rho_crit.value * (box_size / h) ** 3

    # DM particle mass (if baryon simulation) in Msol.
    dm_mass = M_tot_dm / pl_params.n_particles

    # DM particle mass (if DMO simulation) in Msol.
    dm_mass_dmo = M_tot_dm_dmo / pl_params.n_particles

    # Gas particle mass (if baryon simulation) in Msol.
    gas_mass = M_tot_gas / pl_params.n_particles

    print(
        "Dark matter particle mass (if DMO): %.5g Msol (%.10e 1e10 Msol/h)"
        % (dm_mass_dmo, dm_mass_dmo * h / 1.0e10)
    )
    print(
        "Dark matter particle mass: %.5g Msol (%.10e 1e10 Msol/h)"
        % (dm_mass, dm_mass * h / 1.0e10)
    )
    print(
        "Gas particle mass: %.5g Msol (%.10e 1e10 Msol/h)"
        % (gas_mass, gas_mass * h / 1.0e10)
    )

    # Store for later.
    pl_params.total_box_mass = M_tot_dm_dmo
    pl_params.gas_particle_mass = gas_mass


def compute_softening(pl_params):
    """
    Compute softning legnths of the particles.

    Parameters
    ----------
    pl_params : PLParams object

    Adds to pl_params object
    ------------------------
    eps_dm : float
        Dark matter co-moving softening length in Mpc/h
    eps_dm_physical : float
        Max physical dark matter softening length in Mpc/h
    eps_baryon : float
        Gas co-moving softening length in Mpc/h
    eps_baryon_physical : float
        Max physical gas softening length in Mpc/h
    """

    # What softening rules to use?
    if pl_params.softening_rules == "flamingo":
        # Flamingo.
        raise NotImplementedError
        comoving_ratio = 1 / 25.0
        physical_ratio = 1 / 100.0  # Transition at z=3.
    elif pl_params.softening_rules == "eagle-xl":
        # EagleXL.
        raise NotImplementedError
        comoving_ratio = 1 / 20.0
        physical_ratio = 1 / 45.0  # Transition at z = 1.25
    elif pl_params.softening_rules == "eagle":
        # Eagle.
        comoving_ratio = 1 / 25.0
        physical_ratio = 1 / 95.0  # Transition at z = 2.8
    else:
        raise ValueError("Bad softening rule")

    # Cube root of the total number of particles.
    N = pl_params.n_particles ** (1 / 3.0)

    # Mean inter-particle seperation.
    mean_inter = pl_params.box_size / N

    # Dark matter softening lengths.
    pl_params.eps_dm = mean_inter * comoving_ratio
    pl_params.eps_dm_physical = mean_inter * physical_ratio

    # Baryon softening lengths.
    if pl_params.softening_rules == "flamingo":
        pl_params.eps_baryon = pl_paramseps_dm
        pl_params.eps_baryon_physical = pl_params.eps_dm_physical
    elif pl_params.softening_rules == "eagle-xl":
        fac = ((pl_params.Omega0 - pl_params.OmegaBaryon) / pl_params.OmegaBaryon) ** (
            1.0 / 3
        )
        pl_params.eps_baryon = pl_params.eps_dm / fac
        pl_params.eps_baryon_physical = pl_params.eps_dm_physical / fac
    elif pl_params.softening_rules == "eagle":
        pl_params.eps_baryon = pl_params.eps_dm
        pl_params.eps_baryon_physical = pl_params.eps_dm_physical

    print(
        "Comoving Softenings: DM=%.6f Baryons=%.6f Mpc/h"
        % (pl_params.eps_dm, pl_params.eps_baryon)
    )
    print(
        "Max phys Softenings: DM=%.6f Baryons=%.6f Mpc/h"
        % (pl_params.eps_dm_physical, pl_params.eps_baryon_physical)
    )
    print(
        "Comoving Softenings: DM=%.6f Baryons=%.6f Mpc"
        % (
            pl_params.eps_dm / pl_params.HubbleParam,
            pl_params.eps_baryon / pl_params.HubbleParam,
        )
    )
    print(
        "Max phys Softenings: DM=%.6f Baryons=%.6f Mpc"
        % (
            pl_params.eps_dm_physical / pl_params.HubbleParam,
            pl_params.eps_baryon_physical / pl_params.HubbleParam,
        )
    )
