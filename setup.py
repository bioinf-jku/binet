#from distutils.core import setup
from setuptools import setup
from setuptools import Extension
from Cython.Build import cythonize

from codecs import open
from os import path
import numpy as np
VERSION = '2016.03'


# Get the long description from the relevant file
with open(path.join('README.rst'), encoding='utf-8') as f:
    LONG_DESCRIPTION = f.read()


blas_info = np.__config__.get_info('blas_opt_info')

setup(
    name='binet',

    version=VERSION,

    description='Deep Neural Net code for GPUs and CPUs',
    long_description=LONG_DESCRIPTION,
    url='https://github.com/untom/binet',

    # Author details
    author='Thomas Unterthiner',
    author_email='thomas.unterthiner@gmx.net',

    # Choose your license
    license='GPLv2+',

    # See https://pypi.python.org/pypi?%3Aaction=list_classifiers
    classifiers=['Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering',
        'Topic :: Software Development',
        'License :: OSI Approved :: GNU General Public License v2 or later (GPLv2+)',
        'Programming Language :: Python :: 2',
        'Programming Language :: Python :: 2.6',
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.3',
        'Programming Language :: Python :: 3.4',
    ],

    keywords='Machine Learning Deep Learning Neural Nets',

    # You can just specify the packages manually here if your project is
    # simple. Or you can use find_packages().
    packages=['binet'],
    ext_modules = cythonize([Extension("binet.op", ["binet/op.pyx"],
              libraries=blas_info.get('libraries', []),
              extra_link_args=blas_info.get('extra_link_args', []),
              extra_compile_args=blas_info.get('extra_compile_args', []) + ['-Wno-unused-function'],
              define_macros=blas_info.get('define_macros', []),
              library_dirs=blas_info.get('library_dirs', []),
              include_dirs=[np.get_include()] + blas_info.get('include_dirs', []),
              )
    ]),
    setup_requires=["cffi>=1.0.0"],
    cffi_modules=["binet/external_build.py:ffi"],
    install_requires=['cffi>=1.0.0', 'numpy', 'scikit-learn'],
)
