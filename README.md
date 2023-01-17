[![python](https://img.shields.io/badge/Python-3.7-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Zoom-in simulations particle load creator

Script to generate particle loads of "zoom-in" simulations from  [Virgo
Consortium](https://virgo.dur.ac.uk/)-like cosmological parent simulations. This script partners with the [zoom_mask_creator](https://github.com/stuartmcalpine/zoom_mask_creator) script, which generates the input mask to define the Lagrangian region the particle load has to cover.

The particle loads created by `zoom_particle_load_creator` are designed to be fed into `ic_gen`, which generates the actual initial conditions to the simulation codes.

See the example below for more details on what `zoom_particle_load_creator` is doing.


## Installation

### Requirements

* `OpenMPI` or other MPI library
* `python>=3.8,<3.11`

Recommended modules when working on COSMA7:

* `module load gnu_comp/11.1.0 openmpi/4.1.4 parallel_hdf5/1.12.0 python/3.9.1-C7`

### Installation from source

It is recommended you install the package within a virtual/conda environment.
Or alternatively, if you are installing on a shared access machine, to your
local profile by including the `--user` flag during the `pip` installation. You can ofcourse also install directly into your base Python environment if you prefer.

First make sure your `pip` is up-to-date:

* `python3 -m pip install --upgrade pip`

Then you can install the `zoom-particle-load-creator` package by typing the following in
the git directory: 

* `python3 -m pip install -e .`

which will install `zoom-particle-load-creator` and any dependencies.

As a final step, compile the `Cython` dependencies. Goto `./src/particle_load/cython/` and run `cythonize -i MakeGrid.pyx`.

### MPI installation for `read_swift`

If you are using `read_swift` and want to load large snapshots over MPI collectively
(i.e., multiple cores read in parallel from the same file), a bit of additional
setup is required.

Make sure you have MPI libraries installed on your machine (`OpenMPI` for example), and you have `hdf5` installed with **parallel** compatibility ([see here for details](https://docs.h5py.org/en/stable/mpi.html)).

First, uninstall any installed versions of `mpi4py` and `h5py`:

* `python3 -m pip uninstall mpi4py h5py`

Then reinstall `mpi4py` and `h5py` from source with MPI flags:

* `MPICC=mpicc CC=mpicc HDF5_MPI="ON" python3 -m pip install --no-binary=mpi4py,h5py mpi4py h5py`

If `pip` can't find your `HDF5` libraries automatically, e.g., `error: libhdf5.so: cannot open shared object file: No such file or directory`. You will have to specify the path to the HDF5 installation, i.e., `HDF5_DIR=/path/to/hdf5/lib` (see [here](https://docs.h5py.org/en/stable/build.html#building-against-parallel-hdf5) for more details).

For our COSMA7 setup, that will be:

`HDF5DIR="/cosma/local/parallel-hdf5//gnu_11.1.0_ompi_4.1.4/1.12.0/"`

## Usage

Once installed the `zoom-particle-load-creator` command will be available, e.g.,

* `zoom-particle-load-creator --args ./examples/Eagle_Standard_Res.yml`

or in MPI:

* `mpirun -np XX zoom-particle-load-creator --args ./examples/Eagle_Standard_Res.yml`

### Command line arguments

`zoom-particle-load-creator` always requires a parameter file as the final argument. What `zoom-particle-load-creator` outputs depends on the optional command line arguments chosen:

| Argument | Alt Argument | Description |
| --- | ----------- | ------------|
| No arguments | - | `zoom-particle-load-creator` generates a summary of the PL it would create using this parameter file (useful for a quick check before writing any data) |
| `-PL` | `--save_pl_data` | Save the particle load as Fortan binary files compatible with `ic_gen` to the `output_dir` folder |
| `-PL_HDF5` | `--save_pl_data_hdf5` | Save the particle load as a HDF5 to the `output_dir` folder |
| `-IC` | `--make_ic_gen_param_files` | Autogenerate a `ic_gen` parameter file and submission script for the generated particle load (put in the `output_dir` folder) |
| `-S` | `--make_swift_param_files` | Autogenerate a `Swift` parameter file and submission script for the generated particle load (put in the `output_dir` folder) |
| `-MPI` | `--with-mpi` | Flag to use if your are running with MPI |

### Parameter file

All the parameters of the run are stored in a single YAML file, also see ./examples/parameter_list.yml for a full of all the parameters available, here we just list the minimum required parameters `zoom-particle-load-creator` expects in the parameter file:

| Parameter (always required) | Description |
| --- | ----------- |
| `box_size` |  Size of the parent volume in `Mpc/h` |
| `n_particles` | Total number of particles that would fill the parent volume at the target resolution |
| `glass_num` | How many particles are in a glass cell? (must be a cube root of `n_particles` |
| `save_dir` | Where to save the particle load and template files to |

| Parameter (if running with `--make_ic_gen_param_files`) | Description |
| --- | ----------- |
| `panphasian_descriptor` | Panphasia descriptor string |
| `ndim_fft_start` | The number after the `S` in the `panphasian_descriptor` |
| `is_zoom` | Is this a zoom simulation? (True/False) |
| `Omega0` | Value of Omega0 |
| `OmegaCDM` | Value of OmegaCDM |
| `OmegaLambda` | Value of OmegaLambda |
| `OmegaBaryon` | Value of OmegaBaryon |
| `HubbleParam` | Value of HubbleParam |
| `Sigma8` | Value of Sigma8 |
| `linear_ps` | Path to linear power spectrum input into `ic_gen` |

| Parameter (if running with `--make_swift_param_files`) | Description |
| --- | ----------- |
| `softening_rules` | How to compute softening values ("eagle") |

### An example: Group 100 from the Eagle 100 Mpc cosmological simulation

<p float="left">
  <img src="/docs/5Mpc_1_high_res.png" width="400" />
  <img src="/docs/5Mpc_1_low_res_skins.png" width="400" /> 
</p>

Continuing on from the example in [zoom_mask_creator](https://github.com/stuartmcalpine/zoom_mask_creator).

The `Eagle_Standard_Res.yml` parameter file in the `./examples/` directory will generate a particle load that resimulates the 100th largest group from the Eagle 100 Mpc cosmological simulation. As input, we require the HDF5 mask generated from the [zoom_mask_creator](https://github.com/stuartmcalpine/zoom_mask_creator) example.

To get this to run you will need `ic_gen` version `8` or `8.4` installed.

The images above show the outputted particle load generated by this example. Zoom-in particle loads are generated in two main parts. First, the target high-resolution region is built from a rubix cube-like stacking of glass files. A cubic grid goes over the target Lagrangian region (defined by the bounds of our mask). Then, the regions within the high-res grid that overlap with the mask are filled with target resolution glass particles. The remainder of the cells in the high-res grid are filled with ever decreasing numbers of glass particles as you get further from the centre of the high-res region. The left hand plot shows this for our example, in white is the target Lagrangian region, and from blue to yellow, the cells are filled with increasingly lower resolution glass particles.

Then, stage 2, is to fill the remainder of the parent volume. This is done by surrounding the high-res region with ever decreasing resolution shells until you reach the edge of the box, shown in the right hand figure.

Much of the information in the parameter file is to produce `ic_gen` and `Swift` parameter files and submission scripts.


