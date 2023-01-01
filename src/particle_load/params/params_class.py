import particle_load.mympi as mympi
import numpy as np
import yaml
import h5py

from particle_load.params import compute_masses, compute_softening


class ZoomRegionMask:
    def __init__(self, mask_file):
        """
        Load the high-resolution region mask file.

        Parameters
        ----------
        mask_file : string
            Path to the mask HDF5 file

        Attributes
        ----------
        coords : ndarray[N, 3]
            Mask cell coordinates
        geo_centre : ndarray[1,3]
            Centre point of the mask relative to the simulation box
        bounding_length : float
            Bounding box size of the mask
        high_res_volume : float
            Total volume of the high res region
        grid_cell_width : float
            Mask cell size
        """

        mympi.print_section_header("Zoom mask file")
        f = h5py.File(mask_file, "r")
        self.coords = np.array(f["Coordinates"][...], dtype="f8")
        self.geo_centre = f["Coordinates"].attrs.get("geo_centre")
        self.bounding_length = f["Coordinates"].attrs.get("bounding_length")
        self.high_res_volume = f["Coordinates"].attrs.get("high_res_volume")
        self.grid_cell_width = f["Coordinates"].attrs.get("grid_cell_width")
        f.close()
        print("Loaded: %s" % mask_file)
        print("Mask bounding length = %s Mpc/h" % self.bounding_length)


class ParticleLoadParams:
    def __init__(self, args):
        """
        Stores the parameter file entries.

        Defaults are filled in for parameters missing from the input parameter
        file.

        For a full list of parameter options and to see their description, see
        ./examples/parameter_list.yml.

        Each parameter from the parameter file (or defaults) is stored with the
        same name in this class, e.g., X.box_size.

        Parameters
        ----------
        args : argparse object
            Has the command line arguments.
        """

        # These must be in the parameter file, others are optional.
        self.required_params = [
            "box_size",
            "n_particles",
            "glass_num",
            "save_dir",
            "panphasian_descriptor",
            "ndim_fft_start",
            "is_zoom",
            "Omega0",
            "OmegaCDM",
            "OmegaLambda",
            "OmegaBaryon",
            "HubbleParam",
            "Sigma8",
            "linear_ps",
            "softening_rules",
        ]

        # These **cannot** be in param file, they are command line args.
        self.arg_list = [
            "make_ic_gen_param_files",
            "make_swift_param_files",
            "save_pl_data",
            "save_pl_data_hdf5",
            "with_mpi",
        ]

        # Deal with command line arguments.
        self._read_args(args)

        # Read the parameter file.
        self._read_param_file()

        # Fill in default values.
        self._populate_defaults()

        # Compute some numbers based on the input.
        self._compute_numbers()

        # Sanity checks.
        self._sanity_checks()

        # Set cosmological parameters.
        self._set_cosmology()

        # Load zoom region mask file.
        if self.mask_file is not None:
            self._load_high_res_region_mask()

    def _load_high_res_region_mask(self):
        """Load the mask HDF5 file."""

        self.high_res_region_mask = ZoomRegionMask(self.mask_file)

        # Overwrite the high-res region coordinates using details from the mask file.
        self.coords = self.high_res_region_mask.geo_centre
        print(f"Set coordinates of high res region using mask: {self.coords}")

    def _read_args(self, args):
        """Parse command line arguments."""

        # Path to parameter file.
        self.param_file = args.param_file

        mympi.print_section_header("Command line args")
        for att in self.arg_list:
            setattr(self, att, getattr(args, att))
            print(f"{att}: \033[92m{getattr(args, att)}\033[0m")

    def _sanity_checks(self):
        """Perform some basic sanity checks on params we have."""

        # Make sure coords is a numpy array.
        if not isinstance(self.coords, np.ndarray):
            self.coords = np.array(self.coords)

        # Check particle numbers add up.
        assert (
            np.true_divide(self.n_particles, self.glass_num) % 1 < 1e-6
        ), "Number of particles must divide into glass_num"

        assert (
            self.nL_glass_cells_whole_volume**3 * self.glass_num == self.n_particles
        ), "Number of particles must divide into glass_num 2"

        # Check we have the constraints descriptors.
        assert self.num_constraint_files <= 2
        if self.num_constraint_files > 0:
            for i in range(self.num_constraint_files):
                assert hasattr(self, f"constraint_phase_descriptor_{i+1}")
                assert hasattr(self, f"constraint_phase_descriptor_{i+1}_path")
                assert hasattr(self, f"constraint_phase_descriptor_{i+1}_levels")

        # Want to make ic files, have we said the path?
        if self.make_ic_gen_param_files:
            assert hasattr(self, "ic_gen_template_set")
            assert hasattr(self, "ic_gen_exec")

        # Want to make SWIFT files, have we said the path?
        if self.make_swift_param_files:
            assert hasattr(self, "swift_ic_dir_loc")
            assert hasattr(self, "swift_template_set")
            assert hasattr(self, "swift_exec")

        # For non-zoom simulations.
        if self.is_zoom == False:
            assert self.n_species == 1, "Must be 1, not a zoom"
            assert self.multigrid_ics == 0, "Must be 0, not a zoom"
            assert hasattr(self, f"glass_file_loc")

        # If selecting a mask file, make sure its a zoom simulation.
        if self.mask_file is not None:
            assert self.is_zoom, "Mask file must use is_zoom"

        # Can't have more than 2 species.
        assert self.n_species >= 1 and self.n_species <= 2

    def _set_cosmology(self):
        """Set cosmological params."""

        mympi.print_section_header("Masses and Softenings")
        # Compute the mass of the volume given the cosmology.
        compute_masses(self)

        # Compute the softening lengths.
        compute_softening(self)

    def _read_param_file(self):
        """Read the particle load parameter YAML file."""

        with open(self.param_file) as f:
            data = yaml.load(f, Loader=yaml.FullLoader)

        # Make sure we have the minimum params.
        for att in self.required_params:
            assert att in data.keys(), f"{att} is a required parameter."

        # Make sure these are not in param file.
        for att in self.arg_list:
            assert att not in data.keys(), f"Command line arg {att} cannot be a param."

        # Print and store read params.
        mympi.print_section_header("Parameter file")

        print(f"Loaded {len(data)} parameters from {self.param_file}:\n")
        for att in data.keys():
            print(f" - \033[96m{att}\033[0m: \033[92m{data[att]}\033[0m")
            setattr(self, att, data[att])

    def _populate_defaults(self):
        """Fill in the default values (won't overwrite passed params)."""

        self._add_default_value("coords", np.array([0.0, 0.0, 0.0]))
        self._add_default_value("radius", 0.0)
        self._add_default_value("radius_factor", 1.0)
        self._add_default_value("is_slab", False)

        self._add_default_value("mask_file", None)
        self._add_default_value("num_constraint_files", 0)
        self._add_default_value("nq_mass_reduce_factor", 1 / 2.0)
        self._add_default_value("skin_reduce_factor", 1 / 8.0)
        self._add_default_value("min_num_per_cell", 8)
        self._add_default_value("glass_buffer_cells", 2)
        self._add_default_value("ic_region_buffer_frac", 1.0)
        self._add_default_value("starting_z", 127.0)
        self._add_default_value("finishing_z", 0.0)
        self._add_default_value("nmaxpart", 36045928)
        self._add_default_value("nmaxdisp", 791048437)
        self._add_default_value("mem_per_core", 18.2e9)
        self._add_default_value("max_particles_per_ic_file", 400**3)
        self._add_default_value("use_ph_ids", True)
        self._add_default_value("nbit", 21)
        self._add_default_value("fft_times_fac", 2.0)
        self._add_default_value("multigrid_ics", False)
        self._add_default_value("min_nq", 20)
        self._add_default_value("_max_nq", 1000)
        self._add_default_value("grid_also_glass", True)
        self._add_default_value("glass_files_dir", "./glass_files/")
        self._add_default_value("softening_ratio_background", 0.02)
        self._add_default_value("ncores_node", 28)
        
        # to doc
        self._add_default_value("n_nodes_swift", 1)
        self._add_default_value("num_hours_swift", 10)
        self._add_default_value("swift_exec_location", ".")
        self._add_default_value("num_hours_ic_gen", 10)
        self._add_default_value("n_cores_ic_gen", None)
        self._add_default_value("verbose", True)
        self._add_default_value("high_res_L", 0.0)
        self._add_default_value("high_res_n_eff", 0)
        self._add_default_value("import_file", 0)
        self._add_default_value("pl_rep_factor", 1)

    def _add_default_value(self, att, value):
        """Add default parameter value to data array."""

        if not hasattr(self, att):
            setattr(self, att, value)

    def _compute_numbers(self):
        """Some computations based on parameter file."""

        # (Cube root of) how many glass cells would fill the whole simulation volume.
        self.nL_glass_cells_whole_volume = int(
            np.rint((self.n_particles / self.glass_num) ** (1 / 3.0))
        )

        # Size of a glass cell in Mpc/h.
        self.size_glass_cell_mpch = np.true_divide(
            self.box_size, self.nL_glass_cells_whole_volume
        )
