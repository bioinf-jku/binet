from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import nose
import copy
import numpy as np
from nose.tools import assert_raises
from numpy.testing import (assert_allclose, assert_array_equal,
                          assert_almost_equal)

from binet import op
import pycuda.gpuarray as gpuarray

from binet import gpucsrarray
from binet.gpucsrarray import GPUCSRArray


op.init_gpu()
#gpucsrarray.init()


from nose.tools import nottest



def test_sparseB_sgemm():
    from scipy.sparse import csr_matrix

    A = np.random.randn(4, 5).astype(np.float32)
    B = np.random.laplace(size=(4, 6)).astype(np.float32)
    B[(B < 0.1)] = 0
    B = csr_matrix(B, dtype=np.float32)
    X = np.ones((5, 6), dtype=np.float32)
    X_exp =(A.T * B) + X
    op.add_dot(A, B, X, transA=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmB transA=True")

    A = np.random.randn(5, 4).astype(np.float32)
    B = np.random.laplace(size=(4, 6)).astype(np.float32)
    B[(B < 0.1)] = 0
    B = csr_matrix(B, dtype=np.float32)
    X = np.ones((5, 6), dtype=np.float32)
    X_exp =(A * B) + X
    op.add_dot(A, B, X, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmB transA=False")

    A = np.random.randn(5, 4).astype(np.float32)
    B = np.random.laplace(size=(6, 4)).astype(np.float32)
    B[(B < 0.1)] = 0
    B = csr_matrix(B, dtype=np.float32)
    X = np.ones((5, 6), dtype=np.float32)
    X_exp =(A * B.T) + X
    op.add_dot(A, B, X, transB=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmB transA=False, transB=True")

    A = np.random.randn(4, 5).astype(np.float32)
    B = np.random.laplace(size=(6, 4)).astype(np.float32)
    B[(B < 0.1)] = 0
    B = csr_matrix(B, dtype=np.float32)
    X = np.ones((5, 6), dtype=np.float32)
    X_exp =(A.T * B.T) + X
    op.add_dot(A, B, X, transA=True, transB=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmB transA=True, transB=True")


def test_sparseA_sgemm():
    from scipy.sparse import csr_matrix

    A = np.random.laplace(size=(4, 6)).astype(np.float32)
    A[(A < 0.1)] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.randn(6, 5).astype(np.float32)
    X = np.ones((4, 5), dtype=np.float32)
    X_exp =(A * B) + X
    op.add_dot(A, B, X, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmA")

    A = np.random.laplace(size=(6, 4)).astype(np.float32)
    A[(A < 0.1)] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.randn(6, 5).astype(np.float32)
    X = np.ones((4, 5), dtype=np.float32)
    X_exp =(A.T * B) + X
    op.add_dot(A, B, X, transA=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmA transA")

def test_sparseA2_sgemm():
    from scipy.sparse import csr_matrix
    A = np.random.laplace(size=(4, 6)).astype(np.float32)
    A[(A < 0.1)] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.randn(5, 6).astype(np.float32)
    X = np.ones((4, 5), dtype=np.float32)
    X_exp =(A * B.T) + X
    op.add_dot(A, B, X, transB=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmA transB")

    A = np.random.laplace(size=(6, 4)).astype(np.float32)
    A[(A < 0.1)] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.randn(5, 6).astype(np.float32)
    X = np.ones((4, 5), dtype=np.float32)
    X_exp =(A.T * B.T) + X
    op.add_dot(A, B, X, transA=True, transB=True, beta=1.0)
    assert_allclose(X, X_exp, rtol=1e-4, err_msg="sparse_sgemmA transA transB")


def test_gpusparseA_sgemm():
    from scipy.sparse import csr_matrix
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32, order='c')

    X_exp = (A*B) + 0.5*C
    Ad = GPUCSRArray(A)
    Bd = op.to_gpu(B)
    Cd = op.to_gpu(C)
    Xd = op.add_dot(Ad, Bd, Cd, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="gpusparse_sgemm")


def test_gpusparseB_sgemm_ta():
    from scipy.sparse import csr_matrix
    B = np.random.laplace(size=(5, 3)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    A = np.random.normal(size=(5, 4)).astype(np.float32, order="c")
    C = np.ones((A.shape[1], B.shape[1]), dtype=np.float32, order='c')
    X_exp = (A.T*B) + 0.5*C

    Bd = GPUCSRArray(B)
    Ad = op.to_gpu(A)
    Cd = op.to_gpu(C)
    Xd = op.add_dot(Ad, Bd, Cd, transA=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="gpusparse_sgemmB_ta")


def test_gpusparseB_sgemm_tb():
    from scipy.sparse import csr_matrix
    B = np.random.laplace(size=(3, 5)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    A = np.random.normal(size=(4, 5)).astype(np.float32, order="c")
    C = np.ones((A.shape[0], B.shape[0]), dtype=np.float32, order='c')
    X_exp = (A*B.T) + 0.5*C

    Bd = GPUCSRArray(B)
    Ad = op.to_gpu(A)
    Cd = op.to_gpu(C)
    Xd = op.add_dot(Ad, Bd, Cd, transB=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-4, err_msg="gpusparse_sgemmB tb")


def tes_deactivate_t_gpusparseB_sgemm_ta_bug():
    from scipy.sparse import csr_matrix
    A = np.random.normal(size=(6, 12)).astype(np.float32, order="c")
    B = np.random.laplace(size=(6, 33)).astype(np.float32)
    B[B<0.1] = 0
    B = csr_matrix(B, dtype=np.float32)
    C = np.ones((12, 33), dtype=np.float32, order='c')
    X_exp = (A.T*B) + 0.5*C

    Bd = GPUCSRArray(B)
    Ad = op.to_gpu(A)
    Cd = op.to_gpu(C)
    Xd = op.add_dot(Ad, Bd, Cd, transA=True, alpha=1.0, beta=0.5)
    assert_allclose(Xd.get(), X_exp, rtol=1e-3, err_msg="gpusparse_sgemmB ta bug")

@nottest
def test_csrmm_bug():
    ''' the 2nd call might crash'''
    from scipy.sparse import csr_matrix
    W = np.random.normal(size=(5, 3)).astype(np.float32, order="c")
    X = np.random.laplace(size=(6, 3)).astype(np.float32)
    X[X<0.1] = 0
    X = csr_matrix(X, dtype=np.float32)

    Xd = GPUCSRArray(X)
    Wd = op.to_gpu(W)
    Cd = op.dot(Xd, Wd, False, True, out=None, stream=op.streams[0])
    op.add_dot(Cd, Xd, Wd, True, False, alpha=-0.3, beta=1.0, stream=op.streams[0])
    op.mean(Cd, axis=0, stream=op.streams[1])
