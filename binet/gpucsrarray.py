# -*- coding: utf-8 -*-
'''
CSR Array implementation for PyCUDA

Copyright © 2013-2015 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

import numpy as np
import pycuda.gpuarray as gpuarray
from binet import cusparse

from scipy import sparse
from pycuda.driver import memcpy_dtod, mem_alloc

from skcuda import cublas
cublas_handle = None
cusparse_handle = None

def init():
    import atexit

    def _shutdown_gpucsrarray():
        cublas.cublasDestroy(cublas_handle)
        cusparse.cusparseDestroy(cusparse_handle)

    global cublas_handle, cusparse_handle
    if cublas_handle is None or cusparse_handle is None:
        cublas_handle = cublas.cublasCreate()
        cusparse_handle =  cusparse.cusparseCreate()
        atexit.register(_shutdown_gpucsrarray)


class GPUCSRArray(object):
    """A GPUArrayCSR is used to do sparse array-based calculation on the GPU.

    This is mostly supposed to be a numpy-workalike. Operators
    work on an element-by-element basis, just like numpy.ndarray.
    """

    def __init__(self, array, dtype=None, allocator=mem_alloc, stream=None):

        self.dtype = array.dtype if dtype is None else dtype
        self.nnz = array.nnz
        self.shape = array.shape

        if self.nnz == 0:  # let's not waste time
            return

        if not sparse.isspmatrix_csr(array):
            array = sparse.csr_matrix(array, dtype=self.dtype)

        if not array.has_sorted_indices:
            array = array.sorted_indices()

        if stream is not None:
            self.data = gpuarray.to_gpu_async(array.data.astype(dtype=self.dtype), allocator=allocator, stream=stream)
            self.indptr = gpuarray.to_gpu_async(array.indptr, allocator=allocator, stream=stream)
            self.indices = gpuarray.to_gpu_async(array.indices, allocator=allocator, stream=stream)
        else:
            self.data = gpuarray.to_gpu(array.data.astype(dtype=self.dtype), allocator=allocator)
            self.indptr = gpuarray.to_gpu(array.indptr, allocator=allocator)
            self.indices = gpuarray.to_gpu(array.indices, allocator=allocator)
        self.descr = cusparse.cusparseCreateMatDescr()

    def __del__(self):
        if self.nnz > 0:
            cusparse.cusparseDestroyMatDescr(self.descr)

    def get(self):
        assert(self.nnz > 0)
        data = self.data.get()
        indptr = self.indptr.get()
        indices = self.indices.get()
        return sparse.csr_matrix((data, indices, indptr), dtype=self.dtype)

    def todense(self, out=None, allocator=mem_alloc, stream=None):
        if out is None:
            out = gpuarray.empty(self.shape, allocator=allocator, dtype=self.dtype, order="C")

        if self.nnz == 0:  # weird but happens
            out.fill(0.0, stream=stream)
            return out

        # we need to out-of-place transpose if we want rowmajor outputs
        # thus we need a temporary to store our results
        if out.flags.c_contiguous:
            tmp = gpuarray.empty(self.shape, allocator=allocator, dtype=self.dtype, order="C")
        else:
            tmp = out

        if stream is not None:
            cusparse.cusparseSetStream(cusparse_handle, stream.handle)
            cublas.cublasSetStream(cublas_handle, stream.handle)

        cusparse.cusparseScsr2dense(cusparse_handle, self.shape[0],
            self.shape[1], self.descr, self.data.gpudata, self.indptr.gpudata,
            self.indices.gpudata, tmp.gpudata, tmp.shape[0])

        if out.flags.c_contiguous:
            cublas.cublasSgeam(cublas_handle, 1, 1, tmp.shape[1], tmp.shape[0],
                           1.0, tmp.gpudata, tmp.shape[0],
                           0.0, 0, tmp.shape[0], out.gpudata, out.shape[1])
        if stream is not None:
            cusparse.cusparseSetStream(cusparse_handle, 0)
            cublas.cublasSetStream(cublas_handle, 0)

        return out


__cuda_matrix_cache = {}
def __cuda_get_temp_matrix(shape, dtype):
    try:
        x = __cuda_matrix_cache[shape]
    except KeyError:
        x = gpuarray.empty(shape, dtype=dtype)
        __cuda_matrix_cache[shape] = x
    return x


def csrmm2(A_gpu, B_gpu, C_gpu, transA=False, transB=False, alpha=1.0, beta=0.0):
    ''' Calculates C += alpha * A*B + beta*C.
        Where A is sparse and both A and B can be transposed.
    '''

    if transA:
        ta = cusparse.CUSPARSE_OPERATION_TRANSPOSE
        n, l = A_gpu.shape[1], A_gpu.shape[0]
    else:
        ta = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        n, l = A_gpu.shape

    if (B_gpu.flags.c_contiguous and transB) or (B_gpu.flags.f_contiguous and not transB):
        tb = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
    else:
        tb = cusparse.CUSPARSE_OPERATION_TRANSPOSE

    k, m = (B_gpu.shape[1], B_gpu.shape[0]) if transB else B_gpu.shape

    assert (l == k) and (n, m) == C_gpu.shape

    ldb = B_gpu.shape[1] if B_gpu.flags.c_contiguous else B_gpu.shape[0]
    ldc = C_gpu.shape[0]

    # if C-major, save result into a temp array and transpose afterwards
    if C_gpu.flags.c_contiguous:
        out = __cuda_get_temp_matrix(C_gpu.shape, C_gpu.dtype)
        if beta != 0.0:
           memcpy_dtod(out.gpudata, C_gpu.gpudata, C_gpu.nbytes)
    else:
        out = C_gpu

    cusparse.cusparseScsrmm2(cusparse_handle, ta, tb,
        n, m, k, A_gpu.nnz, alpha,
        A_gpu.descr, A_gpu.data.gpudata, A_gpu.indptr.gpudata, A_gpu.indices.gpudata,
        B_gpu.gpudata, ldb, beta, out.gpudata, ldc)

    if C_gpu.flags.c_contiguous:
        cublas.cublasSgeam(cublas_handle, 1, 1, m, n,
                           1.0, out.gpudata, C_gpu.shape[0],
                           0.0, 0, C_gpu.shape[0], C_gpu.gpudata, C_gpu.shape[1])
    return C_gpu


def csrmmB(A_gpu, B_gpu, C_gpu, transA=False, transB=False, alpha=1.0, beta=0.0):
    ''' Calculates C += alpha * A*B + beta*C.
        Where B is sparse and both A and B can be transposed.

        Note: cuSPARSE only allows for sparse A, so we need some tricks:
            Essentially, we will compute C^T = B^T * A^T
            By enforcing C to be row-major, can drop its transpose
            since cuSPARSE assumes column-major. Thus, we only need to
            compute
            C = op(B)^T * op(A)^T
    '''
    assert C_gpu.flags.c_contiguous
    m, k = B_gpu.shape
    ta = cusparse.CUSPARSE_OPERATION_TRANSPOSE if not transB else cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE

    if transA:
        if ta:  # we can't have ta and tb true at the same time according to cuSPARSE docs
            out = __cuda_get_temp_matrix(A_gpu.shape, A_gpu.dtype)
            cublas.cublasSgeam(cublas_handle, 1, 1, A_gpu.shape[0], A_gpu.shape[1], 1.0, A_gpu.gpudata, A_gpu.shape[1],
                               0.0, A_gpu.gpudata, A_gpu.shape[1], out.gpudata, A_gpu.shape[0])
            out.shape = A_gpu.shape[1], A_gpu.shape[0]
            out.strides = gpuarray._c_contiguous_strides(out.dtype.itemsize, out.shape)
            A_gpu = out
            tb = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
            n = A_gpu.shape[0]
        else:
            tb = cusparse.CUSPARSE_OPERATION_TRANSPOSE
            n = A_gpu.shape[1]
    else:
        tb = cusparse.CUSPARSE_OPERATION_NON_TRANSPOSE
        n = A_gpu.shape[0]

    ldb = A_gpu.shape[1]
    ldc = C_gpu.shape[1]

    cusparse.cusparseScsrmm2(cusparse_handle, ta, tb,
        m, n, k, B_gpu.nnz, alpha,
        B_gpu.descr, B_gpu.data.gpudata, B_gpu.indptr.gpudata, B_gpu.indices.gpudata,
        A_gpu.gpudata, ldb, beta, C_gpu.gpudata, ldc)
    return C_gpu
