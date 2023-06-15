from setuptools import setup
from Cython.Build import cythonize
import numpy

setup(
    ext_modules = cythonize("./src/particle_load/cython/MakeGrid.pyx"),
    include_dirs=[numpy.get_include()]
)
