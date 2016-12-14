from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
from Cython.Distutils import build_ext
import numpy as np

cmdclass = {}
ext_modules = []

cmdclass.update({
    'build_ext': build_ext
})

ext_modules += [
    Extension(
        'spark.matrix',
        sources=['spark/matrix.pyx', 'spark/matrix_c.c'],
        include_dirs=[np.get_include()]
    )
]

setup(
    name = 'spark',
    version = '1.0.0',
    packages = find_packages(exclude=['tf-cnn/*']),
    install_requires = ['numpy', 'cython'],
    cmdclass = cmdclass,
    ext_modules = cythonize(ext_modules)
)
