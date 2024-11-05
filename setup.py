import numpy as np
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize

def numpy_include():
    try:
        numpy_include = np.get_include()
    except AttributeError:
        numpy_include = np.get_numpy_include()
    return numpy_include


ext_modules = [
    Extension(
        'reid.CEval.rank_cylib.rank_cy',
        ['reid/CEval/rank_cylib/rank_cy.pyx'],
        include_dirs=[numpy_include()],
    )
]
__version__ = '1.0.0'

setup(
    name='APL',
    version='1.0.0',
    description='Lifelong Person Re-Identification by Adaptive Perspicacity Learning',
    author='Jinze Huang',
    license='MIT',
    packages=find_packages(),
    keywords=['Person Re-Identification', 'Deep Learning', 'Computer Vision'],
    ext_modules=cythonize(ext_modules)
)