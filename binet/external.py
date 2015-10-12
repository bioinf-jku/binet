# -*- coding: utf-8 -*-
'''
Wrapper functions that call external functionality

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''


from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import os
import numpy as np
from scipy import sparse
from ._external import ffi, lib

def sample_gaussian(out, mu, sigma, seed):
    pout = ffi.cast("float*", out.ctypes.data)
    lib.sampleGaussian(pout, out.size, mu, sigma, seed)
    return out


def sample_uniform(out, a, b, seed):
    pout = ffi.cast("float*", out.ctypes.data)
    lib.sampleUniform(pout, out.size, a, b, seed)
    return out


def __csrmm_impl(a, b, c, m, k, n, ldb, ldc, alpha, beta, ta, isFortranOrder):
    if ffi is None:
        raise RuntimeError("MKL not available")
    pm = ffi.new('int[]', [m])
    pk = ffi.new('int[]', [k])
    pn = ffi.new('int[]', [n])
    palpha = ffi.new('float[]', [alpha])
    pval = ffi.cast("float*", a.data.ctypes.data)
    if isFortranOrder:
        matdescr = ffi.new("char[]", "GGNF".encode("ascii"))
        a.indices += 1
        a.indptr -= 1
    else:
        matdescr = ffi.new("char[]", "GGNC".encode("ascii"))
    pindx = ffi.cast("int*", a.indices.ctypes.data)
    pntrb = ffi.cast("int*", a.indptr.ctypes.data)
    pntre = ffi.cast("int*", a.indptr.ctypes.data)
    pntre += 1
    pb = ffi.cast("float*", b.ctypes.data)
    pldb = ffi.new('int[]', [ldb])
    pbeta = ffi.new('float[]', [beta])
    pldc = ffi.new('int[]', [ldc])
    pc = ffi.cast("float*", c.ctypes.data)
    lib.mkl_scsrmm(ta, pm, pn, pk, palpha, matdescr, pval, pindx, pntrb, pntre, pb, pldb, pbeta, pc, pldc)
    if isFortranOrder:
        a.indices -= 1
        a.indptr += 1


def csrmm(a, b, c, transA=False, transB=False, alpha=1.0, beta=0.0):
    assert a.dtype == b.dtype
    if len(a.shape) < 2:
        a = a.reshape(1, a.shape[0])
    if len(b.shape) < 2:
        b = b.reshape(1, b.shape[0])
    m, k = b.shape if transB else (b.shape[1], b.shape[0])
    l, n = a.shape if transA else (a.shape[1], a.shape[0])
    assert c.shape == (n, m) and c.dtype == a.dtype
    assert a.dtype == np.float32 and b.dtype == np.float32
    assert c.flags.c_contiguous

    if a.dtype == np.float32:
        alpha = np.float32(alpha)
        beta = np.float32(beta)
    elif a.dtype == np.float64:
        alpha = np.float64(alpha)
        beta = np.float64(beta)

    if sparse.isspmatrix_csr(a):
        ldb = b.shape[1]
        ta = 't'.encode("ascii") if transA else 'n'.encode("ascii")
        m, k = a.shape
        if not transB:
            l, n = b.shape
            ldc = c.shape[1]
            __csrmm_impl(a, b, c, m, k, n, ldb, ldc, alpha, beta, ta, False)
        else:
            n, l = b.shape
            tmp = c.astype(dtype=c.dtype, order="f")
            ldc = c.shape[0]
            __csrmm_impl(a, b, tmp, m, k, n, ldb, ldc, alpha, beta, ta, True)
            c[:] = tmp[:]
    elif sparse.isspmatrix_csr(b):
        # MKL offers only Y += op(B)*A       (with B sparse)
        # but our call is Y += op(A)*op(B)     (with B sparse)
        # We will use calculate  (op(B)^T*op(A)^T)^T, using the fortran ("one-based")
        # version of the call. Since Y is row-major, we can ignore the outer
        # transpose. We will have to transpose A manually, though
        assert not sparse.issparse(a)
        if transA:
            a = a.astype(dtype=np.float32, order='F')  # transpose a
        m, k = b.shape
        l, n = a.shape if transA else (a.shape[1], a.shape[0])
        ldb = l
        ldc = c.shape[1]
        ta = 'n'.encode("ascii") if transB else 't'.encode("ascii")
        __csrmm_impl(b, a, c, m, k, n, ldb, ldc, alpha, beta, ta, True)
    return c
