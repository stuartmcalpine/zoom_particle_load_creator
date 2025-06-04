from setuptools import setup, Extension
from Cython.Build import cythonize
import numpy

extensions = [
    Extension(
        "particle_load.cython.MakeGrid",  # Name of the generated shared object (.so)
        ["./src/particle_load/cython/MakeGrid.pyx"],
        include_dirs=[numpy.get_include()],
        define_macros=[('NPY_NO_DEPRECATED_API', 'NPY_1_7_API_VERSION')],
    )
]

setup(
    ext_modules = cythonize(extensions),
)
