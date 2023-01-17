[![python](https://img.shields.io/badge/Python-3.7-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.8-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.9-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![python](https://img.shields.io/badge/Python-3.10-3776AB.svg?style=flat&logo=python&logoColor=white)](https://www.python.org)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

## Zoom-in simulations particle load creator

<p float="left">
  <img src="/docs/5Mpc_1_high_res.png" width="400" />
  <img src="/docs/5Mpc_1_low_res_skins.png" width="400" /> 
</p>


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

`zoom-particle-load-creator` always requires a parameter file as the final argument. What `zoom-particle-load-creator` outputs depends on the optional command line arguments:

| Argument | Alt Argument | Description |
| --- | ----------- | ------------|
| No arguments | - | `zoom-particle-load-creator` generates a summary of the PL it would create using this parameter file (useful for a quick check before writing any data) |
| `-PL` | `--save_pl_data` | Save the particle load as Fortan binary files compatible with `ic_gen` to the `output_dir` folder |
| `-PL_HDF5` | `--save_pl_data_hdf5` | Save the particle load as a HDF5 to the `output_dir` folder |
| `-IC` | `--make_ic_gen_param_files` | Autogenerate a `ic_gen` parameter file and submission script for the generated particle load (put in the `output_dir` folder) |
| `-S` | `--make_swift_param_files` | Autogenerate a `Swift` parameter file and submission script for the generated particle load (put in the `output_dir` folder) |
| `-MPI` | `--with-mpi` | Flag to use if your are running with MPI |
