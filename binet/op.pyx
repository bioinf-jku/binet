# -*- coding: utf-8 -*-
'''
GPU/CPU wrapper functions, based on PyCUDA.

Cythonized because otherwise the speed-difference for very small matrices
(e.g. LSTM for addition problem) on CPU is very slow.

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)

Note: this code re-uses code snippes that originate from cudamat, which is
Copyright (c) 2009-2013, Volodymyr Mnih.
See  https://github.com/cudamat/cudamat/blob/d1f9a583c7552dce9ce8747e72aa7a437f1c51ed/LICENSE
for the cudamat license.
'''

from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

cimport cython
cimport numpy as np
from binet cimport blas

import numpy as np
import scipy as sp
import pandas as pd
import inspect
import os
import time
import warnings
import logging


class _DummyStream:
    def synchronize(self):
        pass

_NSTREAMS = 16
streams = [_DummyStream()] * _NSTREAMS

from scipy.sparse.sparsetools import csc_matvecs, csr_matvecs


__cuda_sampler = None
__cuda_swap_rows_kernel = None
__cuda_softmax = None
cuda_memory_pool = None
cuda_hostmemory_pool = None


_IS_CUDA_INITIALIZED = False
__pycuda_context = None
__pycuda_device = None


try:
    import pycuda.gpuarray as gpuarray
    import pycuda.curandom as curand
    import pycuda.driver as drv
    from pycuda.curandom import XORWOWRandomNumberGenerator
    from pycuda.elementwise import ElementwiseKernel
    from pycuda import cumath
    from pycuda import elementwise
    from pycuda.compiler import SourceModule
    from pycuda.driver import Stream
    from pycuda.driver import memcpy_dtod_async
    from skcuda import linalg
    from skcuda import misc as skcuda_misc
    from skcuda.cublas import cublasSgemm, cublasDgemm, cublasSaxpy, \
                              cublasDaxpy, cublasSetStream
    from pycuda.tools import DeviceMemoryPool, PageLockedMemoryPool
    from binet import gpucsrarray
    from binet.gpucsrarray import GPUCSRArray, csrmm2, csrmmB

    __cuda_multiply, __cuda_relu, __cuda_sigmoid, __cuda_crossentropy_core, \
    __cuda_toplayer_delta, __cuda_drelu_delta, __cuda_dtanh_delta, \
    __cuda_dsigmoid_delta, __cuda_randomly_replace_elements, __cuda_l1reg, \
    __cuda_soft_threshold, __cuda_clip, __cuda_swapaxes01, \
    __cuda_sequence_to_tensor, __cuda_to_onehot, __cuda_leakyrelu, \
    __cuda_dleakyrelu_delta, __cuda_mse_core , \
    __cuda_elu, __cuda_delu_delta = [None] * 20

except ImportError:
    warnings.warn("CUDA libraries are not available.")

    # dummy class so typechecks work
    class gpuarray:
        class GPUArray: pass

    # dummy class so typechecks work
    class cumath:
        class log: pass
        class exp: pass
        class tanh: pass
        class fabs: pass
        class sqrt: pass

    class GPUCSRArray: pass


__np_sampler = None
__np_saxpy = None
__EPS = float(np.finfo(np.float32).tiny)


_has_external = False
try:
    from . import external
    _has_external = True
except ImportError:
    warnings.warn("External modules not available")
    pass


def set_seed(seed = 0):
    global __np_sampler, __cuda_sampler, _IS_CUDA_INITIALIZED
    if seed is 0:
        seed = np.uint32(hash(os.getpid() + time.time()) % 4294967295)
    __np_sampler = np.random.RandomState(seed)
    np.random.seed(seed)

    if _IS_CUDA_INITIALIZED:
        s = lambda N: gpuarray.to_gpu(np.array(N * [seed], dtype=np.int32))
        __cuda_sampler = XORWOWRandomNumberGenerator(s)


set_seed() # default-initialize RNG


def shutdown():
    global _IS_CUDA_INITIALIZED, __pycuda_context, __pycuda_device
    if _IS_CUDA_INITIALIZED:
        import skcuda.misc
        from pycuda.tools import clear_context_caches
        cuda_memory_pool.stop_holding()
        cuda_hostmemory_pool.stop_holding()
        skcuda.misc.shutdown()
        clear_context_caches()
        __pycuda_context.pop()
        __pycuda_context = None
        __pycuda_device = None
        import gc
        gc.collect()
    _IS_CUDA_INITIALIZED = False


def init_gpu(gpu_id=0, seed=0):
    '''NOTE: resets seed if it was already set! '''
    global cuda_hostmemory_pool, __cuda_swapaxes01, \
        __cuda_swap_rows_kernel, streams, __cuda_softmax, cuda_memory_pool, \
        __cuda_multiply, __cuda_relu, __cuda_sigmoid, __cuda_crossentropy_core, \
        __cuda_toplayer_delta, __cuda_drelu_delta, __cuda_dtanh_delta, \
        __cuda_dsigmoid_delta, __cuda_randomly_replace_elements, __cuda_l1reg, \
        __cuda_soft_threshold, __cuda_clip, _IS_CUDA_INITIALIZED, \
        __cuda_sequence_to_tensor, __cuda_to_onehot, \
        __cuda_leakyrelu, __cuda_dleakyrelu_delta, __cuda_mse_core, \
        __cuda_elu, __cuda_delu_delta

    if _IS_CUDA_INITIALIZED:
        warnings.warn("GPU was already initialized, will not initialize again!")
        return

    import pycuda.driver as cuda
    global __pycuda_context, __pycuda_device
    cuda.init()
    __pycuda_device = cuda.Device(gpu_id)
    __pycuda_context = __pycuda_device.make_context()
    import atexit
    atexit.register(shutdown)
    import skcuda.misc

    skcuda.misc.init()
    cuda_memory_pool = DeviceMemoryPool()
    cuda_hostmemory_pool = PageLockedMemoryPool()
    gpucsrarray.init()
    _IS_CUDA_INITIALIZED = True



    __cuda_leakyrelu =  ElementwiseKernel("float* x, float* o, float beta",
        "o[i] = x[i] > 0 ? x[i] : beta*x[i];", 'leaky_relu_eltw')
    __cuda_dleakyrelu_delta = ElementwiseKernel("float* d, float* a, float beta",
        "d[i] *= (a[i] > 0 ? 1.0 : beta);", 'dleaky_relu_eltw')

    __cuda_multiply = ElementwiseKernel('float* x, float alpha, float* y',
                                        'y[i] = alpha * x[i];',
                                        'eltw_mult')
    __cuda_relu = ElementwiseKernel("float* x, float* y",
                                    "y[i] = fmaxf(x[i], 0.0f);",
                                    'eltw_relu')
    __cuda_sigmoid = ElementwiseKernel("float* x, float* y",
                                       "y[i] = 1.f / (1.f + __expf(-x[i]));",
                                       'eltw_sigmoid')
    __cuda_crossentropy_core = ElementwiseKernel(
        "float* target, float* pred, float* out",
        "out[i] = target[i] == target[i] ? target[i] * __logf(pred[i] + 1e-16) : 0;", 'eltw_ce_core')

    __cuda_mse_core = ElementwiseKernel(
        "float* target, float* pred, float* out",
        '''
        float d = target[i] - pred[i];
        out[i] = d == d ? d*d : 0.0;
        ''',
        'eltw_mse_core')

    __cuda_toplayer_delta = ElementwiseKernel(
        "float* a, float* y, float* out",
        "out[i] = (y[i] == y[i]) ? (a[i] - y[i]) : 0.0f;", 'toplayer_delta')
    __cuda_drelu_delta = ElementwiseKernel(
        "float* d, float* a", "d[i] *= (a[i] > 0)", 'drelu_delta')
    __cuda_dtanh_delta =ElementwiseKernel(
        "float* d, float* a", "d[i] *= (1.0 - a[i]*a[i])", 'dtanh_delta')
    __cuda_dsigmoid_delta =ElementwiseKernel(
        "float* d, float* a", "d[i] *= a[i]*(1.0 - a[i])", 'dsigmoid_delta')

    __cuda_elu =  ElementwiseKernel("float* x, float* o, float alpha",
        'o[i] = x[i] > 0 ? x[i] : alpha*(expf(x[i])-1);',
        'elu_eltw')
    __cuda_delu_delta = ElementwiseKernel("float* d, float* a, float alpha",
        'd[i] *= (a[i] > 0 ? 1.0 : a[i]+alpha);',
        'delu_eltw')


    # drops "val" into x p times of the time. r contains (0, 1] uniform values.
    # Resulting mask will be stored in r, as well.
    __cuda_randomly_replace_elements = ElementwiseKernel(
        'float* x, float* r, float p, float val',
        '''
           r[i] = r[i] > p;
           x[i] = r[i] ? x[i] : val;
        ''', 'eltwise_rand_replace')
    __cuda_l1reg = ElementwiseKernel(
        'float* w, float* dw, float eta, float l1_penalty, float* target',
        '''
        float s = copysignf(1.0f, w[i]);
        float nw = w[i] + eta*dw[i] - l1_penalty*s;
        target[i] = s* fmax(0, s*nw);
        ''', 'eltw_l1reg')

    __cuda_soft_threshold = ElementwiseKernel(
        'float* x, float alpha, float* target',
        '''
        const float f = x[i];
        target[i] = f > 0 ? fmaxf(0., f - alpha) : fminf(0., f + alpha);
        ''', 'eltw_softthreshold')

    __cuda_clip = ElementwiseKernel(
        'float* x, float minval, float maxval, float* out',
        '''
        float m = x[i] > minval ? x[i] : minval;
        out[i] = m < maxval ? m : maxval;
        ''', 'eltw_clip')

    __cuda_swapaxes01 = ElementwiseKernel(
        'float* x, float* out, unsigned n1, unsigned n2, unsigned s0, unsigned s1, unsigned s2',
        '''
        unsigned c = i % n2;
        unsigned b = (i / n2) % n1;
        unsigned a = i / (n2*n1);
        unsigned iidx = c*s2+b*s1 + a*s0;
        out[i] = x[iidx];
        ''', 'swapaxes01')


    __cuda_sequence_to_tensor = ElementwiseKernel(
        "float* out, unsigned short* sequence, unsigned seqlen, unsigned n_samples, unsigned n_classes",
        """
        unsigned cid = i % n_classes;
        unsigned sid = (i / n_classes) % n_samples;
        unsigned tid = i / (n_classes * n_samples);
        out[tid*(n_classes*n_samples) + sid*n_classes + cid] = sequence[sid+tid] == cid;
        """, 'seq_to_tensor')

    # assumes array has been zeroed out before
    __cuda_to_onehot = ElementwiseKernel(
        "float* out, short* labels, unsigned n_samples, unsigned n_classes",
        """
        unsigned cid = i % n_classes;
        unsigned sid = i / n_classes;
        out[sid*n_classes + cid] = labels[sid] == cid ? 1.0 : 0.0;
        """, 'to_onehot')


    streams = []
    for _ in range(_NSTREAMS):
        streams.append(Stream())

    # the following kernels are taken from cudamat / hebel
    code = """
    #include "float.h"
    __global__ void shuffleRows(float* X, int32_t* target_idx,
                        float* Xout, const int32_t n, const int32_t m)
        {
            const int tid = blockIdx.x * blockDim.x + threadIdx.x;
            if (tid >= n)
                return;
            const int idx = target_idx[tid];
            for (int i = 0; i < m; ++i)
                Xout[tid*m + i] = X[idx*m + i];
            //target_idx[tid] = tid;
        }

    __global__ void ksoftmax(float* mat,
                             float* tmp,
                             float* out,
                             unsigned int height,
                             unsigned int width) {
          __shared__ float max_vals[32];
        float cur_max = -FLT_MAX;
        float val = 0;

        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            val = mat[blockIdx.x * width + i];
            if (val > cur_max)
                cur_max = val;
        }

        max_vals[threadIdx.x] = cur_max;
        __syncthreads();
        if (threadIdx.x == 0) {
            cur_max = -FLT_MAX;
            for (unsigned int i = 0; i < 32; i++) {
                if (max_vals[i] > cur_max)
                    cur_max = max_vals[i];
            }
            tmp[blockIdx.x] = cur_max;
        }
        __syncthreads();


        float sum = 0.0;
        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            float x =  __expf(mat[blockIdx.x * width + i] - tmp[blockIdx.x]);
            out[blockIdx.x * width + i] = x;
            sum += x;
        }
        max_vals[threadIdx.x] = sum;
        __syncthreads();
        if (threadIdx.x == 0) {
            sum = 0.0;
            for (unsigned int i = 0; i < 32; i++)
                sum += max_vals[i];
            tmp[blockIdx.x] = sum;
        }
        __syncthreads();
        for (unsigned int i = threadIdx.x; i < width; i += 32) {
            out[blockIdx.x * width + i] /= tmp[blockIdx.x];
        }
    }
    """

    mod = SourceModule(code)
    __cuda_swap_rows_kernel = mod.get_function('shuffleRows')
    __cuda_softmax = mod.get_function("ksoftmax")
    set_seed(seed)


cdef __cuda_swap_rows(X, idx, Xout):
    assert X.flags.c_contiguous
    assert X.shape == Xout.shape
    n = X.shape[0]
    block = (32, 1, 1)
    gridx = n // block[0] + 1 * (n % block[0] != 0)
    grid = (gridx, 1, 1)
    __cuda_swap_rows_kernel(X, idx, Xout, np.int32(n),
                            np.int32(X.shape[1]), block=block, grid=grid)

cpdef to_gpu(x, stream=None, dtype=None):
    ''' Transforms all np.ndarray fields to the pycuda.gpuarray (as float32!).'''
    if isinstance(x, logging.Logger):
        return x
    elif isinstance(x, pd.DataFrame): # this seems to crash otherwise
        return x
    elif type(x) == np.ndarray:
        dt = x.dtype if dtype is None else dtype

        alloc = cuda_memory_pool.allocate
        if stream is not None:
            x = gpuarray.to_gpu_async(x.astype(dt), allocator=alloc, stream=stream)
        else:
            x = gpuarray.to_gpu(x.astype(dt), allocator=alloc)
    elif type(x) == list:
        for idx, obj in enumerate(x):
            x[idx] = to_gpu(obj)
    # for tuples, we have to construct a temp-list first reconvert later
    elif (type(x) == tuple):
        tmplist = []
        for idx, obj in enumerate(x):
            tmplist.append(to_gpu(obj))
        x = tuple(tmplist)
    elif type(x) == dict:
        for k, obj in x.items():
            x[k] = to_gpu(obj)
    elif inspect.isclass(x) or hasattr(x, '__dict__'):
        for k, obj in x.__dict__.items():
            if not k.startswith("__") and not k.endswith("__"): # python internals
                x.__dict__[k] = to_gpu(obj)
    return x


cpdef to_cpu(x, stream=None):
    ''' Copies all gpuarray.garrays to np.ndarrays.'''
    if isinstance(x, logging.Logger):
        return x
    if isinstance(x, pd.DataFrame): # this seems to crash otherwise
        return x
    elif type(x) == gpuarray.GPUArray:
        if stream is not None:
            return x.get_async(stream=stream)
        else:
            return x.get()
    elif type(x) == list:
        for idx, obj in enumerate(x):
            x[idx] = to_cpu(obj)
    # for tuples, we have to construct a temp-list first
    # and reconvert to tuple afterwards
    elif (type(x) == tuple):
        tmplist = []
        for idx, obj in enumerate(x):
            tmplist.append(to_cpu(obj))
        return tuple(tmplist)
    elif inspect.isclass(x) or hasattr(x, '__dict__'):
        for k, obj in x.__dict__.items():
            if not k.startswith("__") and not k.endswith("__"): # python internals
                x.__dict__[k] = to_cpu(obj)
    return x


cpdef is_on_gpu(x):
    return isinstance(x, gpuarray.GPUArray)


cpdef empty(shape, dtype=np.float32, use_gpu=False):
    if use_gpu:
        return gpuarray.empty(shape, allocator=cuda_memory_pool.allocate,
                              dtype=dtype)
    else:
        return np.empty(shape, dtype=dtype)


cpdef empty_like(X):
    # We don't use empty_like so we are sure to use cuda_memory_pool
    # We don't use np.empty_like so this also works for CSR matrices
    return empty(X.shape, X.dtype, use_gpu=type(X) == gpuarray.GPUArray)


cpdef zeros(shape, dtype=np.float32, use_gpu=False):
    if use_gpu:
        out = gpuarray.empty(shape, allocator=cuda_memory_pool.allocate,
                              dtype=dtype)
        out.fill(0)
    else:
        return np.zeros(shape, dtype=dtype)


cpdef zeros_like(X):
    # We don't use empty_like so we are sure to use cuda_memory_pool
    # We don't use np.empty_like so this also works for CSR matrices
    return zeros(X.shape, X.dtype, use_gpu=type(X) == gpuarray.GPUArray)



def maximum(a, b):
    if isinstance(a, gpuarray.GPUArray):
        return gpuarray.maximum(a, b)
    else:
        return np.maximum(a, b)

def minimum(a, b):
    if isinstance(a, gpuarray.GPUArray):
        return gpuarray.minimum(a, b)
    else:
        return np.minimum(a, b)


def make_function(npfunc, gpufunc):
    def f(x, out=None, stream=None):
        if out is None:
            out = empty_like(x)
        if isinstance(x, gpuarray.GPUArray):
            gpufunc(x, out=out, stream=stream)
            return out
        else:
            return npfunc(x, out=out)
    return f


log = make_function(np.log, cumath.log)
exp = make_function(np.exp, cumath.exp)
tanh = make_function(np.tanh, cumath.tanh)
abs = make_function(np.abs, cumath.fabs)
sqrt = make_function(np.sqrt, cumath.sqrt)


def sigmoid(x, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_sigmoid(x, out, stream=stream)
    else:
        out = np.exp(-x, out=out)
        out += 1.0
        out = np.divide(1.0, out, out=out)
    return out


def relu(x, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_relu(x, out, stream=stream)
        return out
    else:
        return np.maximum(x, 0, out=out)


cpdef max(x, axis=None):
    if isinstance(x, gpuarray.GPUArray):
        return skcuda_misc.max(x, axis)
    else:
        return np.max(x, axis=axis)


def argmax(x, axis=None, stream=None):
    if isinstance(x, gpuarray.GPUArray):
        if axis is None:
            raise NotImplementedError("Can't do global argmax on GPU")
        else:
            return skcuda_misc.argmax(x, axis)
    else:
        return np.argmax(x, axis=axis)


def sum(x, axis=None, stream=None):
    if isinstance(x, gpuarray.GPUArray):
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, stream.handle)
        retval = skcuda_misc.sum(x, axis=axis)
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, 0)
        return retval
    else:
        return np.sum(x, axis=axis)


def mean(x, axis=None, stream=None):
    if isinstance(x, gpuarray.GPUArray):
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, stream.handle)
        retval = skcuda_misc.mean(x, axis=axis)
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, 0)
        return retval
    else:
        return np.mean(x, axis=axis)


def std(X, axis=None, stream=None):
    if isinstance(X, gpuarray.GPUArray):
        return skcuda_misc.std(X, axis=axis, stream=stream)
    else:
        return np.std(X, axis=axis)


def softmax(X, out=None, stream=None):
    if out is None:
        out = empty_like(X)
    if isinstance(X, gpuarray.GPUArray):
        n, m = X.shape
        tmp = empty((1, n), dtype=X.dtype, use_gpu=True)
        __cuda_softmax(X.gpudata, tmp.gpudata, out.gpudata, np.int32(n),
            np.int32(m), stream=stream, block=(32, 1, 1), grid=(n, 1, 1))
        return out
    else:
        m = X.max(axis=1).reshape((-1, 1))
        e = np.exp(X - m, out=out)
        e /= e.sum(axis=1).reshape((-1, 1))
        return e


def toplayer_delta(A, Y, X, stream=None):
    """Calculates the derivative of the error function.
    This assumes that the output-activation function is the canonical
    link function of the error.

    It is possible that some labels are unknown and encoded as NaNs
    (e.g. when doing multi-task learning this happens some times).
    These will get an error backflow of 0.

    NOTE: We're not allowed to change Y, so we need to allocate new result!
    """
    out = empty_like(Y)
    if isinstance(Y, gpuarray.GPUArray):
        __cuda_toplayer_delta(A, Y, out, stream=stream)
    else:
        np.subtract(A, Y, out=out)
        out[Y != Y] = 0.0
    return out


def drelu_delta(D, A, X, stream=None):
    """ Calculates D *= (a > 0)"""
    if isinstance(D, gpuarray.GPUArray):
        __cuda_drelu_delta(D, A, stream=stream)
    else:
        D *= (A > 0)
    return D


def dtanh_delta(D, A, X, stream=None):
    """ Calculates D *= (A > 0)"""
    if isinstance(D, gpuarray.GPUArray):
        __cuda_dtanh_delta(D, A, stream=stream)
    else:
        D *= (1.0-A*A)
    return D


def dsigmoid_delta(D, A, X, stream=None):
    """ Calculates D *= (A > 0)"""
    if isinstance(D, gpuarray.GPUArray):
        __cuda_dsigmoid_delta(D, A, stream=stream)
    else:
        D *= A*(1.0 - A)
    return D


def identity(x, dummy=None, dummy2=None, out=None, stream=None):
    return x


def dlinear_delta(D, A, X, stream=None):
    """ Calculates D *= 1"""
    return D


def cross_entropy(target, pred, stream=None):
    if sp.sparse.issparse(target):
        return to_cpu(-sum(target.multiply(log(pred+__EPS)))) / target.shape[0]
    elif isinstance(target, gpuarray.GPUArray):
        out = empty_like(target)
        __cuda_crossentropy_core(target, pred, out, stream=stream)
        return -gpuarray.sum(out, stream=stream).get() / target.shape[0]
    else:
        out = target * log(pred+__EPS)
        out[target != target] = 0  # deal with NaNs in target
        return to_cpu(-sum(out)) / target.shape[0]


def mean_squared_error(target, pred, stream=None):
    assert not sp.sparse.issparse(target)
    # NOTE: does not deal with NaNs right now!
    if isinstance(target, gpuarray.GPUArray):
        out = empty_like(target)
        __cuda_mse_core(target, pred, out, stream=stream)
        return 0.5*gpuarray.sum(out, stream=stream).get() / target.shape[0]
    else:
        out = target - pred
        out*=out
        out[target != target] = 0  # deal with NaNs in target
        return 0.5*out.mean(0).sum()


def clip(X, minval, maxval, out=None):
    if out is None:
        out = empty_like(X)
    if isinstance(X, gpuarray.GPUArray):
        __cuda_clip(X, minval, maxval, out)
    else:
        out[:] = np.clip(X, minval, maxval, out=out)
    return out


def leakyrelu(x, beta=0.1, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_leakyrelu(x, out, beta, stream=stream)
    else:
        out[:] = np.where(x > 0, x, beta*x)
    return out


def dleakyrelu_delta(D, A, X, beta=0.1, stream=None):
    """ Calculates D *= (a > 0)"""
    if isinstance(D, gpuarray.GPUArray):
        __cuda_dleakyrelu_delta(D, A, np.float32(beta), stream=stream)
    else:
        D *= np.where(A > 0, 1, beta)
    return D


def elu(x, alpha=1.0, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_elu(x, out, alpha, stream=stream)
    else:
        out[:] = np.where(x > 0, x, alpha*(np.exp(x)-1))
    return out


def delu_delta(D, A, X, alpha=1.0, stream=None):
    """ Calculates D *= (a > 0)"""
    if isinstance(D, gpuarray.GPUArray):
        __cuda_delu_delta(D, A, np.float32(alpha), stream=stream)
    else:
        D *= np.where(A > 0, 1, alpha+A)
    return D




######## RANDOM NUMBERS ###########################################
def rand_gaussian(shape, mu=0.0, sigma=1.0, dtype=np.float32, use_gpu=False, stream=None):
    out = empty(shape, dtype, use_gpu)
    rand_gaussian_like(out, mu, sigma, out=out, stream=stream)
    return out


def rand_uniform(shape, a=0.0, b=1.0, dtype=np.float32, use_gpu=False, stream=None):
    out = empty(shape, dtype, use_gpu)
    rand_uniform_like(out, a, b, out=out, stream=stream)
    return out


cpdef rand_int():
    ''' Returns a single random integer number. '''
    return __np_sampler.randint(np.iinfo(np.int32).max)


cpdef rand_gaussian_like(x, mu=0.0, sigma=1.0, out=None, stream=None):
    ''' Generates Gaussian distributed values.

    This is an optimization for generating np.float32 random numbers without
    going through np.random.normal(...).astype(), since that takes a lot of
    time generating doubles (and potentially wasts a lot of memory for large
    arrays). If external is available, we can use that to generate the floats
    directly.'''
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_sampler.fill_normal(out, stream=stream)
        if mu != 0 or sigma != 1.0:
            out *= sigma
            out += mu
    elif out.dtype == np.float32 and _has_external:
        out = external.sample_gaussian(out, mu, sigma, rand_int())
    else:
        out[:] = __np_sampler.normal(loc=mu, scale=sigma, size=out.shape).astype(out.dtype)
    return out


cpdef rand_uniform_like(x, a=0.0, b=1.0, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        __cuda_sampler.fill_uniform(out, stream=stream)
        if a != 0.0 or b != 1.0:
            out = out._axpbz(b-a, a, out)
    elif out.dtype == np.float32 and _has_external:
        out = out = external.sample_uniform(out, a, b, rand_int())
    else:
        out[:] = __np_sampler.uniform(a, b, x.shape).astype(out.dtype)
    return out


def sample_binomial(X):
    if isinstance(X, gpuarray.GPUArray):
        return X > rand_uniform_like(X)
    else:
        # this is faster than np.where(x > r, 1.0, 0.0)
        return (X > __np_sampler.uniform(size=X.shape)).astype(X.dtype)


def reorder_rows(A, idx, output=None):
    if type(A) == gpuarray.GPUArray:
        idxd = gpuarray.to_gpu(np.array(idx, dtype=np.int32))
        if output is None:
            output = gpuarray.empty_like(A)
        __cuda_swap_rows(A, idxd, output)
        return output
    else:
        return A[idx]


def shuffle_rows(X, y, output=None, idx=None):
    if idx is None:
        idx = np.arange(X.shape[0])
    __np_sampler.shuffle(idx)
    if type(X) == gpuarray.GPUArray:
        idxd = gpuarray.to_gpu(np.array(idx, dtype=np.int32))
        if output is None:
            Xout = empty(X.shape, X.dtype, use_gpu=True)
            yout = empty(y.shape, y.dtype, use_gpu=True)
        else:
            Xout, yout = output
        __cuda_swap_rows(X, idxd, Xout)
        __cuda_swap_rows(y, idxd, yout)
        return Xout, yout
    else:
        return X[idx], y[idx]


def randomly_replace_elements(X, p, val, stream=None):
    ''' With a probability of p, replace the values of X by val (in-place!)
    Returns the modified X, as well as a mask M which contains
    0 on changed elements and 1 on unchanged ones.'''
    R = rand_uniform_like(X, stream=stream)
    if isinstance(X, gpuarray.GPUArray):
       __cuda_randomly_replace_elements(X, R, p, val, stream=stream)
       return X, R
    else:
        M = R > p
        if val == 0.0:  # dropout as special case (3x speedup!)
            X *= M
        else:
            X[~M.astype(np.bool)] = val
        return X, M


#### DOT PRODUCTS #########################################################
cpdef add_dot(a, b, out=None, int transA=False, int transB=False,
            double alpha=1.0, beta=1.0, stream=None):
    if a.dtype != b.dtype:
        raise RuntimeError("Incompatible dtypes of A and B: %s <> %s" % (str(a.dtype), str(b.dtype)))
    if len(a.shape) < 2:
        a = a.reshape(1, a.shape[0])
    if len(b.shape) < 2:
        b = b.reshape(1, b.shape[0])

    cdef int m, k, l, n
    m, k = (b.shape[0], b.shape[1]) if transB else (b.shape[1], b.shape[0])
    l, n = (a.shape[0], a.shape[1]) if transA else (a.shape[1], a.shape[0])
    assert k == l
    if out is None:
        use_gpu = isinstance(a, gpuarray.GPUArray) or isinstance(b, gpuarray.GPUArray)
        out = empty((n, m), a.dtype, use_gpu=use_gpu)
    else:
        if out.dtype != a.dtype:
            raise RuntimeError("Incompatible dtypes of A and C: %s <> %s" % (str(a.dtype), str(out.dtype)))
        if (out.shape[0], out.shape[1]) != (n, m):
            raise RuntimeError("Incompatible shapes of A and C: %s <> %s" % (str(out.shape), str((n, m))))

    cdef int lda = k if transB else m
    cdef int ldb = n if transA else k
    cdef int ldc = m

    if isinstance(a, gpuarray.GPUArray) and isinstance(b, gpuarray.GPUArray):
        ta = 't' if transA else 'n'
        tb = 't' if transB else 'n'
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, stream.handle)
        if a.dtype == np.float32:
            cublasSgemm(skcuda_misc._global_cublas_handle, tb, ta,
                        m, n, k, alpha, b.gpudata, lda, a.gpudata, ldb,
                        beta, out.gpudata, ldc)
        elif a.dtype == np.float64:
            cublasDgemm(skcuda_misc._global_cublas_handle, tb, ta,
                        m, n, k, alpha, b.gpudata, lda, a.gpudata, ldb,
                        beta, out.gpudata, ldc)
        else:
            assert(False)
        if stream is not None:
            cublasSetStream(skcuda_misc._global_cublas_handle, 0)
    elif type(a) == np.ndarray and type(b) == np.ndarray:
        add_dot_npy(a, b, out, transA, transB, alpha, beta)
    elif sp.sparse.isspmatrix_csr(a):
        assert a.dtype == np.float32 and b.dtype == np.float32
        assert out.flags['C_CONTIGUOUS']
        assert not sp.sparse.issparse(b)
        # Note: we could use external.csrmm, but that isn't actually faster
        bb = b.T if transB else b
        n_vecs = bb.shape[1]
        out *= beta
        if transA:
            N, M = a.shape
            csc_matvecs(M, N, n_vecs, a.indptr, a.indices, a.data / alpha, bb.ravel(), out)
        else:
            M, N = a.shape
            csr_matvecs(M, N, n_vecs, a.indptr, a.indices, a.data / alpha, bb.ravel(), out)
    elif sp.sparse.isspmatrix_csr(b):
        external.csrmm(a, b, out, transA, transB, alpha, beta)
    elif isinstance(a, GPUCSRArray):
        return gpucsrarray.csrmm2(a, b, out, transA, transB, alpha, beta)
    elif isinstance(b, GPUCSRArray):
        return gpucsrarray.csrmmB(a, b, out, transA, transB, alpha, beta)
    else:
        raise RuntimeError("Unsupported types: %s, %s" % (type(a), type(b)))
    return out


cdef void add_dot_npy(np.ndarray a, np.ndarray b, np.ndarray out,
            int transA=False, int transB=False,
            double alpha=1.0, double beta=1.0) nogil:
    cdef int m, k, l, n
    cdef blas.CBLAS_TRANSPOSE ta, tb

    if transB:
        m, k, tb = b.shape[0], b.shape[1], blas.CblasTrans
    else:
        m, k, tb = b.shape[1], b.shape[0], blas.CblasNoTrans
    if transA:
        l, n, ta = a.shape[0], a.shape[1], blas.CblasTrans
    else:
        l, n, ta = a.shape[1], a.shape[0], blas.CblasNoTrans

    if a.descr.type_num == np.NPY_FLOAT:
        blas.cblas_sgemm(blas.CblasColMajor,tb, ta, m, n, k, <float> alpha,
            <float*>b.data, b.shape[1], <float*>a.data, a.shape[1], <float> beta,
            <float*>out.data, out.shape[1])
    elif a.descr.type_num == np.NPY_DOUBLE:
        blas.cblas_dgemm(blas.CblasColMajor,tb, ta, m, n, k, alpha,
            <double*>b.data, b.shape[1], <double*>a.data, a.shape[1], beta,
            <double*>out.data, out.shape[1])


def dot(a, b, transA=False, transB=False, out=None, alpha=1.0, stream=None):
    return add_dot(a, b, out, transA=transA, transB=transB,
                   alpha=alpha, beta=0.0, stream=stream)


def add_matvec(X, b, axis=None, out=None, stream=None):
    if type(X) == gpuarray.GPUArray:
        return skcuda_misc.add_matvec(X, b, axis=axis, out=out, stream=stream)
    else:
        if axis == 1:
            b = b.T
        elif axis == 0:
            pass
        elif X.shape[0] == b.shape[1]:
            b = b.T
        return np.add(X, b) if out is None else np.add(X, b, out=out)


def mult_matvec(X, b, axis=None, out=None, stream=None):
    if type(X) == gpuarray.GPUArray:
        return skcuda_misc.mult_matvec(X, b, axis=axis, out=out, stream=stream)
    else:
        if axis == 1:
            b = b.T
        elif axis == 0:
            pass
        elif X.shape[0] == b.shape[1]:
            b = b.T
        return np.multiply(X, b) if out is None else np.multiply(X, b, out=out)


def div_matvec(X, b, axis=None, out=None, stream=None):
    if type(X) == gpuarray.GPUArray:
        return skcuda_misc.div_matvec(X, b, axis=axis, out=out, stream=stream)
    else:
        if axis == 1:
            b = b.T
        elif axis == 0:
            pass
        elif X.shape[0] == b.shape[1]:
            b = b.T
        return np.divide(X, b) if out is None else np.divide(X, b, out=out)


# NOTE: both MLK and OpenBLAS (from 0.2.10) know saxpby
#       but this will fail on older OpenBLAS versions
cdef void add_vec_npy(np.ndarray x, double alpha, np.ndarray y, double beta) nogil:
    cdef int i
    cdef int nelems = 1
    for i in range(x.ndim):
        nelems *= x.shape[i]
    if x.descr.type_num == np.NPY_FLOAT:
        blas.cblas_saxpby(nelems, <float>alpha, <float*>y.data, <int>1,
                          <float>(beta), <float*>x.data, <int>1)
    elif x.descr.type_num == np.NPY_DOUBLE:
        blas.cblas_daxpby(nelems, alpha, <double*>y.data, <int>1,
                          beta, <double*>x.data, <int>1)


def add_vec(x, alpha, y, beta=1.0, stream=None):
    '''x := a*y + b*x'''
    assert x.dtype == y.dtype
    if type(x) == gpuarray.GPUArray:
        #pycuda's gpuarray._axpbyz checks shapes, which we don't want:
        assert x.size == y.size
        func = elementwise.get_axpbyz_kernel(x.dtype, y.dtype, x.dtype)
        func.prepared_async_call(x._grid, x._block, stream,
                beta, x.gpudata, alpha, y.gpudata,
                    x.gpudata, x.mem_size)
    else:
        add_vec_npy(x, <double>alpha, y, <double>beta)
    return x


def mult_mm(x, y, out=None, stream=None):
    if out is None:
        out = empty_like(x)
    if isinstance(x, gpuarray.GPUArray):
        x._elwise_multiply(y, out, stream=stream)
    else:
        np.multiply(x, y, out=out)
    return out


def add_vec_l1reg(w, dw, eta, l1_penalty, out=None, stream=None):
    if out is None:
        out = empty_like(w)
    if type(w) == gpuarray.GPUArray:
        __cuda_l1reg(w, dw, eta, l1_penalty, out, stream=stream)
    else:
        # we are not allowed to overshoot 0 because of the prior
        s = np.sign(w)
        nw = w + eta*dw - l1_penalty*s
        return np.multiply(s, np.maximum(0, s*nw, out=out), out=out)
        # slower version:
        #out[:] = np.where(w > 0, np.maximum(0, nw), np.minimum(0, nw))


def soft_threshold(X, alpha, out=None, stream=None):
    if out is None:
        out = empty_like(X)
    if isinstance(X, gpuarray.GPUArray):
        __cuda_soft_threshold(X, alpha, out, stream=stream)
    else:
        np.multiply(np.abs(X) > alpha, X - alpha*np.sign(X), out=out)
    return out


#### MISC ###############################################
def copy(x, stream=None):
    if isinstance(x, gpuarray.GPUArray) and stream is not None:
        new = gpuarray.GPUArray(x.shape, x.dtype)
        memcpy_dtod_async(new.gpudata, x.gpudata, x.nbytes, stream=stream)
        return new
    else:
        return x.copy()


def diag(X):
    if isinstance(X, gpuarray.GPUArray):
        return linalg.diag(X)
    else:
        return np.diag(X)


def inplace_scale_columns(X, scale):
    ''' Scales each column i of X by scale[i].'''
    if isinstance(X, gpuarray.GPUArray):
        linalg.dot_diag(scale, X, 't', overwrite=True)
    else:
        X *= scale[None, :]
    return X


def swapaxes01(X, out=None, stream=None):
    ''' Swaps axis 0 and 1. Assumes 3D array.'''
    n = X.shape
    nout = (n[1], n[0], n[2])
    if out is None:
        out = empty_like(X).reshape(nout)
    if isinstance(X, gpuarray.GPUArray):
        s = X.strides
        isz = X.dtype.itemsize
        __cuda_swapaxes01(X, out, n[0], n[2], s[1]//isz, s[0]//isz, s[2]//isz)
        out = out.reshape(nout)
    else:
        out[:] = np.swapaxes(X, 0, 1)
    return out


#### LSTM FUNCTIONS ###############################################
def to_onehot(labels, unsigned int n_classes, out=None, stream=None):
    cdef unsigned int n_samples = <int>(labels.shape[0])
    assert(labels.dtype == np.uint16)
    cdef int use_gpu = <int>(isinstance(labels, gpuarray.GPUArray))

    if out is None:
        out = empty((n_samples, n_classes), np.float32, use_gpu)
    else:
        assert(out.shape == (n_samples, n_classes))
        assert(out.dtype == np.float32)

    if use_gpu:
        __cuda_to_onehot(out, labels, n_samples, n_classes, stream=stream)
    else:
        # NOTE: on GPUs, it's faster to zero out in the kernel, but on the
        # CPU, using a memset first and then just filling in the labels
        # is faster
        out.fill(0.0)
        to_onehot_npy(labels, n_samples, out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void to_onehot_npy(np.ndarray[np.uint16_t, ndim=1] labels,
                        unsigned int n_samples,
                        np.ndarray[np.float32_t, ndim=2] out):
    '''Transforms a vector of labels into a one-hot representation.

    This approach is around 3x as fast as using
    sklearn.preprocessing.OneHotEncoder
    and consumes a bit less memory.

    NOTE: this function assumes that labels start at 0 and go
    up to ``n_classes``.

    TODO: if we pass different batches into this, we need a way to
          specify what the real number of classes is going to be!
          (Not all classes may be present in the current batch)
    '''
    cdef unsigned int cls, i
    with nogil:
        for i in range(n_samples):
            cls = labels[i]
            out[i, cls] = 1


def sequence_to_tensor(sequence, unsigned int seqlen, unsigned int n_classes,
                       out=None, stream=None):
    '''
    Transforms a sequence of labels into an input block for an LSTM.

    We want to create an input format where, given a small window from the sequence,
    the challenge will be to predict the element following the window.

    Thus, we can produce a series of blocks (of len ``seqlen``), where each
    block (or sample) is just the previous shifted to the left by 1, with the previous
    target appended to the right.

    Note that the very last block will not have a label. To avoid this, we will
    not produce that block.

    Also, we turn the labels into a one-hot encoding and returns a tensor of the form
    (Time x Samples x Features), where time is ``seqlen``, the number of samples/windows
    is dependent on the length of the input sequence and ``Features`` is the one-hot
    encoding of the elements in the sequence (i.e. n_classes)
    '''
    cdef unsigned int n_samples = sequence.shape[0]-seqlen
    cdef int use_gpu = <int>(isinstance(sequence, gpuarray.GPUArray))
    if seqlen > sequence.shape[0]:
        msg = "seqlen is larger than sequence (%d > %d)"
        raise ValueError(msg % (sequence.shape[0], seqlen))
    assert(sequence.dtype == np.uint16)

    if out is None:
        out = empty((seqlen, n_samples, n_classes), np.float32, use_gpu)
    else:
        assert(out.shape == (seqlen, n_samples, n_classes))
        assert(out.dtype == np.float32)
    if use_gpu:
        __cuda_sequence_to_tensor(out, sequence, seqlen, n_samples,
                                  n_classes, stream=stream)
    else:
        out.fill(0.0)
        sequence_to_tensor_npy(sequence, seqlen, n_samples, n_classes, out)
    return out


@cython.boundscheck(False)
@cython.wraparound(False)
cdef void sequence_to_tensor_npy(np.ndarray[np.uint16_t, ndim=1] sequence,
                                 unsigned int seqlen,
                                 unsigned int n_samples,
                                 unsigned int n_classes,
                                 np.ndarray[np.float32_t, ndim=3] out):
    cdef unsigned int t, i, cls
    with nogil:
        for t in range(seqlen):
            for i in range(n_samples):
                cls = sequence[t+i]
                out[t, i, cls] = 1
