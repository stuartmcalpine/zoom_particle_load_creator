from setuptools import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("./src/particle_load/cython/MakeGrid.pyx")
)
