import nose
import copy
import numpy as np

from scipy import sparse
import pycuda.autoinit
import pycuda.gpuarray as gpu
from pycuda.driver import Stream
import binet.cusparse as cusparse

from nose.tools import assert_raises
from numpy.testing import assert_allclose, assert_array_equal


def test_cusparseScsrmm():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = sparse.csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="f")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32)

    X_exp = (A*B) + 0.5*C
    a_data = gpu.to_gpu(A.data)
    a_indptr = gpu.to_gpu(A.indptr)
    a_indices = gpu.to_gpu(A.indices)
    b = gpu.to_gpu(B)

    h = cusparse.cusparseCreate()
    descrA = cusparse.cusparseCreateMatDescr()

    c = gpu.empty((C.shape[1], C.shape[0]), dtype=A.dtype)
    c.fill(1.0)

    cusparse.cusparseScsrmm(h, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
        c.shape[1], c.shape[0], b.shape[0], A.nnz, 1.0,
        descrA, a_data.gpudata, a_indptr.gpudata, a_indices.gpudata,
        b.gpudata, b.shape[0], 0.5, c.gpudata, c.shape[1])
    assert_allclose(c.get().T, X_exp, rtol=1e-4)


def test_cusparseScsrmm2_notranspose():
    A = np.random.laplace(size=(5, 3)).astype(np.float32)
    A[A<0.1] = 0
    A = sparse.csr_matrix(A, dtype=np.float32)
    B = np.random.normal(size=(3, 6)).astype(np.float32, order="f")
    C = np.ones((A.shape[0], B.shape[1]), dtype=np.float32)

    X_exp = (A*B) + 0.5*C
    a_data = gpu.to_gpu(A.data)
    a_indptr = gpu.to_gpu(A.indptr)
    a_indices = gpu.to_gpu(A.indices)
    b = gpu.to_gpu(B)

    h = cusparse.cusparseCreate()
    descrA = cusparse.cusparseCreateMatDescr()

    c = gpu.empty((C.shape[1], C.shape[0]), dtype=A.dtype)
    c.fill(1.0)

    cusparse.cusparseScsrmm2(h, cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
        cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE,
        c.shape[1], c.shape[0], b.shape[0], A.nnz, 1.0,
        descrA, a_data.gpudata, a_indptr.gpudata, a_indices.gpudata,
        b.gpudata, b.shape[0], 0.5, c.gpudata, c.shape[1])
    assert_allclose(c.get().T, X_exp, rtol=1e-4)


def test_cusparseScsr2dense():
    A = np.random.laplace(size=(3, 5)).astype(np.float32)
    A[A<0.1] = 0
    A = sparse.csr_matrix(A, dtype=np.float32)
    A.sort_indices()

    a_data = gpu.to_gpu(A.data)
    a_indptr = gpu.to_gpu(A.indptr)
    a_indices = gpu.to_gpu(A.indices)
    out = gpu.empty((A.shape[0], A.shape[1]), dtype=A.dtype, order="F")


    h = cusparse.cusparseCreate()
    descrA = cusparse.cusparseCreateMatDescr()

    cusparse.cusparseScsr2dense(h, A.shape[0], A.shape[1],
        descrA, a_data.gpudata, a_indptr.gpudata, a_indices.gpudata,
        out.gpudata, out.shape[0])
    assert_allclose(out.get(), A.A, rtol=1e-4)



def test_cusparseSetStream():
    A = np.random.laplace(size=(3, 5)).astype(np.float32)
    A[A<0.1] = 0
    A = sparse.csr_matrix(A, dtype=np.float32)
    A.sort_indices()

    a_data = gpu.to_gpu(A.data)
    a_indptr = gpu.to_gpu(A.indptr)
    a_indices = gpu.to_gpu(A.indices)
    out = gpu.empty((A.shape[0], A.shape[1]), dtype=A.dtype, order="F")


    h = cusparse.cusparseCreate()
    descrA = cusparse.cusparseCreateMatDescr()

    stream = Stream()
    cusparse.cusparseSetStream(h, stream.handle)
    cusparse.cusparseScsr2dense(h, A.shape[0], A.shape[1],
        descrA, a_data.gpudata, a_indptr.gpudata, a_indices.gpudata,
        out.gpudata, out.shape[0])
    cusparse.cusparseSetStream(h, 0)
    stream.synchronize()
    assert_allclose(out.get(), A.A, rtol=1e-4)
