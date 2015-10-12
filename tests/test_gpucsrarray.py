import nose
import copy
import numpy as np

from scipy.sparse import csr_matrix

import pycuda.autoinit
import pycuda.gpuarray as gpu
from pycuda.driver import Stream

from binet import cusparse
from binet.gpucsrarray import *

from numpy.testing import assert_allclose, assert_array_equal

from nose.tools import nottest

init()

def test_gpucsrarray():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = csr_matrix(A, dtype=np.float32)
    Ad = GPUCSRArray(A)
    assert Ad.nnz == A.nnz
    assert Ad.shape == A.shape
    assert Ad.dtype == A.dtype
    assert_allclose(Ad.get().A, A.A, rtol=1e-4)


def test_csrmm2_orderF():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="f")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32, order='f')

    X_exp = (A*B) + 0.5*C
    Ad = GPUCSRArray(A)
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderF")

    B = np.random.normal(size=(6, 3)).astype(np.float32, order="f")
    C = np.ones((A.shape[0], B.shape[0]), dtype=np.float32, order='f')
    X_exp = (A*B.T) + 0.5*C
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderF B.T")


def test_csrmm2_orderC():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32, order='f')

    X_exp = (A*B) + 0.5*C
    Ad = GPUCSRArray(A)
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderC")

    B = np.random.normal(size=(6, 3)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[0]), dtype=np.float32, order='f')
    X_exp = (A*B.T) + 0.5*C
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderC B.T")


def test_csrmm2_orderC2():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32, order='c')

    X_exp = (A*B) + 0.5*C
    Ad = GPUCSRArray(A)
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderC2")

    B = np.random.normal(size=(6, 3)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[0]), dtype=np.float32, order='c')
    X_exp = (A*B.T) + 0.5*C
    Bd = gpu.to_gpu(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmm2(Ad, Bd, Cd, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmm2_orderC2 B.T")


def test_csrmmB():
    A = np.random.normal(size=(4, 5)).astype(np.float32, order="c")
    B = np.random.laplace(size=(5, 3)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32, order='c')
    X_exp = (A*B) + 0.5*C

    Ad = gpu.to_gpu(A)
    Bd = GPUCSRArray(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmmB(Ad, Bd, Cd, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmmB normal")


def test_csrmmB_ta():
    A = np.random.normal(size=(5, 4)).astype(np.float32, order="c")
    B = np.random.laplace(size=(5, 3)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((A.shape[1], B.shape[1]), dtype=np.float32, order='c')
    X_exp = (A.T*B) + 0.5*C

    Ad = gpu.to_gpu(A)
    Bd = GPUCSRArray(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmmB(Ad, Bd, Cd, transA=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="csrmmB tA")


def test_csrmmB_tb():
    A = np.random.normal(size=(4, 5)).astype(np.float32, order="c")
    B = np.random.laplace(size=(3, 5)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((A.shape[0], B.shape[0]), dtype=np.float32, order='c')
    X_exp = (A*B.T) + 0.5*C

    Ad = gpu.to_gpu(A)
    Bd = GPUCSRArray(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmmB(Ad, Bd, Cd, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-3, err_msg="csrmmB tb")


def test_csrmmB_tatb():
    A = np.random.normal(size=(5, 4)).astype(np.float32, order="c")
    B = np.random.laplace(size=(3, 5)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((A.shape[1], B.shape[0]), dtype=np.float32, order='c')
    X_exp = (A.T*B.T) + 0.5*C

    Ad = gpu.to_gpu(A)
    Bd = GPUCSRArray(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmmB(Ad, Bd, Cd, transA=True, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-3, err_msg="csrmmB tatb")



def test_csrmmB_ta_bug():
    ''' Tests for a bug we had previously'''
    A = np.random.normal(size=(5, 14)).astype(np.float32, order="c")
    B = np.random.laplace(size=(5, 3)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((A.shape[1], B.shape[1]), dtype=np.float32, order='c')
    X_exp = (A.T*B) + 0.5*C

    Ad = gpu.to_gpu(A)
    Bd = GPUCSRArray(B)
    Cd = gpu.to_gpu(C)
    Xd = csrmmB(Ad, Bd, Cd, transA=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-3, err_msg="csrmmB tA")


def test_todense():
    ''' Test GPUCSRArray.todense()'''
    X = np.random.laplace(size=(6, 3)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    Xd = GPUCSRArray(X)
    Yd = Xd.todense()
    assert_allclose(Yd.get(), X.A, rtol=1e-3, err_msg="todense")

    # we had a bug with these matrix sizes previously since
    # todense does a transpose
    X = np.random.laplace(size=(193, 65)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    Xd = GPUCSRArray(X)
    Yd = Xd.todense()
    assert_allclose(Yd.get(), X.A, rtol=1e-3, err_msg="todense large")

    X = np.random.laplace(size=(10, 13)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    Xd = GPUCSRArray(X)
    Yd = Xd.todense()
    assert_allclose(Yd.get(), X.A, rtol=1e-3, err_msg="todense fat matrix")

    X = np.random.laplace(size=(10, 13)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    Xd = GPUCSRArray(X)
    Yd = gpu.empty(Xd.shape, dtype=Xd.dtype)
    Yd2 = Xd.todense(out=Yd)
    assert(Yd is Yd2)
    assert_allclose(Yd.get(), X.A, rtol=1e-3, err_msg="todense output")


def test_todense_stream():
    ''' Test GPUCSRArray.todense()'''
    X = np.random.laplace(size=(600, 300)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    Xd = GPUCSRArray(X)

    stream = Stream()

    Yd = Xd.todense(stream=stream)
    stream.synchronize()
    assert_allclose(Yd.get(), X.A, rtol=1e-3, err_msg="todense")


@nottest
def test_csrmm_bug():
    ''' the 2nd call might crash'''
    W = np.random.normal(size=(5, 3)).astype(np.float32, order="c")
    X = np.random.laplace(size=(6, 3)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)
    C = np.empty((X.shape[0], W.shape[0]), dtype=np.float32, order='c')

    Xd = GPUCSRArray(X)
    Wd = gpu.to_gpu(W)
    Cd = gpu.to_gpu(C)
    foo = csrmm2(Xd, Wd, Cd, transB=True, alpha=1.0)
    csrmmB(Cd, Xd, Wd, transA=True, alpha=1.0)
    import gpuwrapper_pycuda as gpuwrapper
    gpuwrapper.init_cuda()
    gpuwrapper.mean(Cd, axis=0)
