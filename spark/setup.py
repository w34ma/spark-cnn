from distutils.core import setup
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np

ext_modules = [
    Extension(
        'matrix',
        sources=['matrix.pyx', 'matrix_c.c'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name = 'matrix',
    ext_modules = cythonize(ext_modules)
)
