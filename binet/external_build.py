# -*- coding: utf-8 -*-
'''
Wrapper functions that call external functionality.
This file holds the CFFI setup code and will only be run during installation

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

import os
from cffi import FFI

# Decides on installation time if MKL or GSL should be used!

ffi = FFI()
mkldir = "/opt/intel/mkl/lib/intel64/"
mkl_incdir = "/opt/intel/mkl/include/"

# some machine specific paths for bioinf@jku machines
if not os.path.exists(mkldir):
    mkldir = "/apps/intel/mkl/lib/intel64/"  # on 'zusie'
    mkl_incdir = "/apps/intel/mkl/include/"  # on 'zusie'
if not os.path.exists(mkldir):
    mkldir = '/system/apps/biosoft/intel/mkl-11.1/mkl/lib/intel64'  # on k40
    mkl_incdir = '/system/apps/biosoft/intel/mkl-11.1/mkl/include'  # on k40
if os.path.exists(mkldir):
    ffi.cdef("""
    void mkl_scsrmm(char *transa, int *m, int *n, int *k,
                    float *alpha, char *matdescra, float *val,
                    int *indx, int *pntrb, int *pntre, float *b,
                    int *ldb, float *beta, float *c, int *ldc);
    void sampleGaussian(float* out, int n, float mu, float sigma, int seed);
    void sampleUniform(float* out, int n, float a, float b, int seed);
    """)

    ffi.set_source('binet._external', r'''
    #include <mkl.h>
    #include <mkl_spblas.h>

    void sampleGaussian(float* out, int n, float mu, float sigma, int seed) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MT19937, seed);
        vsRngGaussian(VSL_METHOD_SGAUSSIAN_ICDF, stream, n, out, mu, sigma);
        vslDeleteStream(&stream);
    }

    void sampleUniform(float* out, int n, float a, float b, int seed) {
        VSLStreamStatePtr stream;
        vslNewStream(&stream, VSL_BRNG_MT19937, seed);
        vsRngUniform(VSL_RNG_METHOD_UNIFORM_STD, stream, n, out, a, b);
        vslDeleteStream(&stream);
    }
    ''', libraries=['mkl_rt'], library_dirs=[mkldir], include_dirs=[mkl_incdir])
else:
    ffi.cdef("""
    void sampleGaussian(float* out, int n, float mu, float sigma, int seed);
    void sampleUniform(float* out, int n, float a, float b, int seed);
    """)

    ffi.set_source('binet._external', r'''
    #include<gsl/gsl_rng.h>
    #include<gsl/gsl_randist.h>

    void sampleGaussian(float* out, int n, float mu, float sigma, int seed) {
        double tmp;
        int i;
        //gsl_rng * r = gsl_rng_alloc (gsl_rng_mt19937);
        gsl_rng * r = gsl_rng_alloc (gsl_rng_taus2);
        gsl_rng_set (r, seed);
        for (i = 0; i < n; ++i){
            tmp = gsl_ran_gaussian_ziggurat(r, sigma);
            out[i] = mu + (float)tmp;
        }
        gsl_rng_free (r);
    }

    void sampleUniform(float* out, int n, float a, float b, int seed) {
        int i;
        double aa = (double) a;
        double bb = (double) b;
        gsl_rng * r = gsl_rng_alloc (gsl_rng_mt19937);
        gsl_rng_set (r, seed);
        for (i = 0; i < n; ++i){
            out[i] = (float) gsl_ran_flat(r, aa, bb);
        }
        gsl_rng_free (r);
    }
    ''', libraries=['gsl', 'blas'])


if __name__ == "__main__":
    ffi.compile()
