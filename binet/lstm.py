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

from pycuda.elementwise import ElementwiseKernel
import numpy as np
from . import op


__cuda_sigma_pi_dtanh = ElementwiseKernel(
    "float* a, float* b, float* f_out, float* out",
    'out[i] = a[i]*b[i]*(1.0f-f_out[i]*f_out[i]);', 'sigma_pi_dtanh')

__cuda_sigma_pi_sigmoid = ElementwiseKernel(
    "float* a, float* b, float* f_out, float* out",
    'out[i] = a[i]*b[i]*f_out[i]*(1.0f-f_out[i]);', 'sigma_pi_dsigmoid')

def sigma_pi_dsigmoid(a, b, f_out, out):
    if isinstance(a, op.gpuarray.GPUArray):
        __cuda_sigma_pi_sigmoid(a, b, f_out, out)
    else:
        np.multiply(a, b, out=out)
        out *= f_out
        out *= (1.0 - f_out)
    return out


def sigma_pi_dtanh(a, b, f_out, out):
    if isinstance(a, op.gpuarray.GPUArray):
        __cuda_sigma_pi_dtanh(a, b, f_out, out)
    else:
        np.multiply(a, b, out=out)
        out *= (1.0 - f_out*f_out)
    return out


__cuda_memorycell_fwd = ElementwiseKernel(
    "float* C, float* prevC, float* Z, float* I, float* F, float* nF, int has_forgetgate, int epoch",
    '''
    if (has_forgetgate) {
        F[i] = 1.f / (1.f + __expf(-nF[i]));
        C[i] = F[i] * prevC[i];
    } else {
        C[i] = epoch > 0 ? prevC[i] : 0.0f;
    }
    C[i] += Z[i] * I[i];
    ''', 'memorycell_fwd')

def memcell_fwd_path(l, i):
    if isinstance(l.X, op.gpuarray.GPUArray):
        assert l.gatefunc == op.sigmoid
        __cuda_memorycell_fwd(l.C[i], l.C[i-1], l.Z[i], l.I[i], l.F[i], l.nF[i], l.has_forgetgate, i)
    else:
        l.C[i] = l.C[i-1] if i > 0 else 0
        if l.has_forgetgate:
            l.gatefunc(l.nF[i], out=l.F[i])
            l.C[i] *= l.F[i]
        l.C[i] += l.Z[i] * l.I[i]


__cuda_memorycell_bwd = ElementwiseKernel(
    "float* dC, float* nextDc, float* prevC, float* F, float* nextF, float* dF, int has_forgetgate, int epoch, int not_last_epoch",
    '''
    if (has_forgetgate) {
        if (not_last_epoch)
            dC[i] += nextDc[i] * nextF[i];
        if (epoch > 0)
            dF[i] = dC[i] * prevC[i] * F[i]*(1-F[i]);
    } else {
        if (not_last_epoch)
            dC[i] += nextDc[i];
    }
    ''', 'memorycell_bwd')
def memcell_bwd_path(l, i):
    if isinstance(l.X, op.gpuarray.GPUArray):
        assert l.gatefunc == op.sigmoid
        ni = i+1 if i < l.X.shape[0]-1 else 0 # we don't need dC[i+1] in this case
        __cuda_memorycell_bwd(l.dC[i], l.dC[ni], l.C[i-1], l.F[i], l.F[ni], l.dF[i], l.has_forgetgate, i, i < l.X.shape[0]-1)
    else:
        if l.has_forgetgate:
            if i < l.X.shape[0]-1:
                l.dC[i] += l.dC[i+1] * l.F[i+1]
            if i > 0:
                l.dF[i] = l.dgatefunc(l.dC[i], l.C[i-1], l.F[i], l.dF[i])
        elif i < l.X.shape[0]-1:
            l.dC[i] += l.dC[i+1]


class LongShortTermMemoryLayer(object):
    def __init__(self, size, n_inputs, max_seq_len, batch_size, has_forgetgate=True, dtype=np.float32):
        self.has_forgetgate = has_forgetgate
        self.max_seq_len = max_seq_len
        self.size = size
        self.n_inputs = n_inputs
        self.batch_size = batch_size
        self.dtype = dtype
        self.set_activation_function("tanh", "sigmoid")
        self.setup()

    def reset(self):
        self.X = np.zeros((self.max_seq_len, self.batch_size, self.n_inputs), dtype=self.dtype)
        self.Z = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.I = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.O = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.C = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.H = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.Y = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)

        self.nZ = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.nI = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.nO = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)

        self.dZ = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.dI = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.dO = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.dC = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.dH = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
        self.dY = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)

        self.dWz = np.zeros_like(self.Wz)
        self.dWi = np.zeros_like(self.Wi)
        self.dWo = np.zeros_like(self.Wo)
        self.dbz = np.zeros_like(self.bz)
        self.dbi = np.zeros_like(self.bi)
        self.dbo = np.zeros_like(self.bo)
        self.dRz = np.zeros_like(self.Rz)
        self.dRi = np.zeros_like(self.Ri)
        self.dRo = np.zeros_like(self.Ro)

        if self.has_forgetgate:
            self.F = np.empty((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
            self.nF = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
            self.dF = np.zeros((self.max_seq_len, self.batch_size, self.size), dtype=self.dtype)
            self.dWf = np.zeros_like(self.Wf)
            self.dRf = np.zeros_like(self.Rf)
            self.dbf = np.zeros_like(self.bf)
        else:
            self.F = np.empty((self.max_seq_len, 1, 1), dtype=self.dtype)
            self.nF = np.zeros((self.max_seq_len, 1, 1), dtype=self.dtype)
            self.dF = np.zeros((self.max_seq_len, 1, 1), dtype=self.dtype)
            self.dWf = None
            self.dRf = None
            self.dbf = None

    def setup(self):
        ws = (self.size, self.n_inputs)
        rs = (self.size, self.size)
        self.Wz = op.rand_gaussian(ws, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.Rz = op.rand_gaussian(rs, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.Wi = op.rand_gaussian(ws, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.Ri = op.rand_gaussian(rs, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.Wo = op.rand_gaussian(ws, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.Ro = op.rand_gaussian(rs, sigma=0.05, dtype=self.dtype, use_gpu=False)
        self.bz = np.zeros((1, self.size), dtype=self.dtype)
        self.bi = np.zeros((1, self.size), dtype=self.dtype)
        self.bo = np.zeros((1, self.size), dtype=self.dtype)
        if self.has_forgetgate:
            self.Wf = op.rand_gaussian(ws, sigma=0.05, dtype=self.dtype, use_gpu=False)
            self.Rf = op.rand_gaussian(rs, sigma=0.05, dtype=self.dtype, use_gpu=False)
            self.bf = -4.0*np.ones((1, self.size), dtype=self.dtype)
        else:
            self.Wf = None
            self.Rf = None
            self.bf = None

        self.reset()

    def _prepare_input(self, X):
        '''Prepares the input for consumption by the LSTM'''
        return op.swapaxes01(X)  # new shape: Time x Samples x Features

    def fprop(self, X):
        self.X = self._prepare_input(X)

        # the non-recurrent parts (gate & input) can be calculated outside
        # of the loop on the flattened input if time is the outermost dimension
        flat_shape = (self.max_seq_len*self.batch_size, self.n_inputs)
        full_shape = (self.max_seq_len, self.batch_size, self.size)
        out_shape =  (self.max_seq_len*self.batch_size, self.size)
        self.nZ = op.dot(self.X.reshape(flat_shape), self.Wz, transB=True, out=self.nZ.reshape(out_shape))
        self.nZ = op.add_matvec(self.nZ, self.bz, out=self.nZ).reshape(full_shape)
        self.nI = op.dot(self.X.reshape(flat_shape), self.Wi, transB=True, out=self.nI.reshape(out_shape))
        self.nI = op.add_matvec(self.nI, self.bi, out=self.nI).reshape(full_shape)
        self.nO = op.dot(self.X.reshape(flat_shape), self.Wo, transB=True, out=self.nO.reshape(out_shape))
        self.nO = op.add_matvec(self.nO, self.bo, out=self.nO).reshape(full_shape)
        if self.has_forgetgate:
            self.nF = op.dot(self.X.reshape(flat_shape), self.Wf, transB=True, out=self.nF.reshape(out_shape))
            self.nF = op.add_matvec(self.nF, self.bf, out=self.nF).reshape(full_shape)

        for i in range(self.X.shape[0]):
            if i > 0:  # no recurrent connections in the first timestep
                op.add_dot(self.Y[i-1], self.Rz, self.nZ[i], transB=True)
                op.add_dot(self.Y[i-1], self.Ri, self.nI[i], transB=True)
                op.add_dot(self.Y[i-1], self.Ro, self.nO[i], transB=True)
                if self.has_forgetgate:
                    op.add_dot(self.Y[i-1], self.Rf, self.nF[i], transB=True)

            self.inputfunc(self.nZ[i], out=self.Z[i])
            self.gatefunc(self.nI[i], out=self.I[i])
            memcell_fwd_path(self, i)

            self.outputfunc(self.C[i], out=self.H[i])
            self.gatefunc(self.nO[i], out=self.O[i])
            op.mult_mm(self.O[i], self.H[i], out=self.Y[i])
        return self.Y[-1]

    def bprop(self, D, momentum = 0.0):
        self.dWz *= momentum
        self.dWi *= momentum
        self.dWo *= momentum
        self.dbz *= momentum
        self.dbi *= momentum
        self.dbo *= momentum
        self.dRz *= momentum
        self.dRi *= momentum
        self.dRo *= momentum
        if self.has_forgetgate:
            self.dWf *= momentum
            self.dRf *= momentum
            self.dbf *= momentum

        self.dY = D.copy()
        for i in range(D.shape[0]-1, -1, -1):
            if i < D.shape[0]-1:
                op.add_dot(self.dZ[i+1], self.Rz, self.dY[i])
                op.add_dot(self.dI[i+1], self.Ri, self.dY[i])
                op.add_dot(self.dO[i+1], self.Ro, self.dY[i])
                if self.has_forgetgate:
                    op.add_dot(self.dF[i+1], self.Rf, self.dY[i])

            self.dgatefunc(self.dY[i], self.H[i], self.O[i], self.dO[i])
            self.doutputfunc(self.dY[i], self.O[i], self.H[i], self.dC[i])
            memcell_bwd_path(self, i)

            self.dgatefunc(self.dC[i], self.Z[i], self.I[i], self.dI[i])
            self.dinputfunc(self.dC[i], self.I[i], self.Z[i], self.dZ[i])

            alpha = 1.0 / self.batch_size
            self.dWz = op.add_dot(self.dZ[i], self.X[i], out=self.dWz, transA=True, alpha=alpha)
            op.add_vec(self.dbz, 1.0, op.mean(self.dZ[i], 0))
            self.dWi = op.add_dot(self.dI[i], self.X[i], out=self.dWi, transA=True, alpha=alpha)
            op.add_vec(self.dbi, 1.0, op.mean(self.dI[i], 0))
            self.dWo = op.add_dot(self.dO[i], self.X[i], out=self.dWo, transA=True, alpha=alpha)
            op.add_vec(self.dbo, 1.0, op.mean(self.dO[i], 0))
            if self.has_forgetgate and i > 0:
                self.dWo = op.add_dot(self.dF[i], self.X[i], out=self.dWf, transA=True, alpha=alpha)
                op.add_vec(self.dbf, 1.0, op.mean(self.dF[i], 0))

            if i < D.shape[0]-1:
                self.dRz = op.add_dot(self.dZ[i+1], self.Y[i], out=self.dRz, transA=True, alpha=alpha)
                self.dRi = op.add_dot(self.dI[i+1], self.Y[i], out=self.dRi, transA=True, alpha=alpha)
                self.dRo = op.add_dot(self.dO[i+1], self.Y[i], out=self.dRo, transA=True, alpha=alpha)
                if self.has_forgetgate:
                    self.dRo = op.add_dot(self.dF[i+1], self.Y[i], out=self.dRf, transA=True, alpha=alpha)

    def update(self, learning_rate, stream=None):
        op.add_vec(self.Wz, -learning_rate, self.dWz)
        op.add_vec(self.Wi, -learning_rate, self.dWi)
        op.add_vec(self.Wo, -learning_rate, self.dWo)
        op.add_vec(self.bz, -learning_rate, self.dbz)
        op.add_vec(self.bi, -learning_rate, self.dbi)
        op.add_vec(self.bo, -learning_rate, self.dbo)
        op.add_vec(self.Rz, -learning_rate, self.dRz)
        op.add_vec(self.Ri, -learning_rate, self.dRi)
        op.add_vec(self.Ro, -learning_rate, self.dRo)
        if self.has_forgetgate:
            op.add_vec(self.Wf, -learning_rate, self.dWf)
            op.add_vec(self.bf, -learning_rate, self.dbf)
            op.add_vec(self.Rf, -learning_rate, self.dRf)


    def set_activation_function(self, inputfunction, outputfunction):
        self.inputfunction=inputfunction
        self.outputfunction=outputfunction
        assert outputfunction == "sigmoid"  # TODO: implement switching
        assert inputfunction == "tanh"  # TODO: implement switching

        self.gatefunc = op.sigmoid
        self.dgatefunc = sigma_pi_dsigmoid
        self.inputfunc = op.tanh
        self.dinputfunc = sigma_pi_dtanh
        self.outputfunc = op.sigmoid
        self.doutputfunc = sigma_pi_dsigmoid


    def __getstate__(self):
        weightstate = [self.Wz, self.Wi, self.Wo, self.Wf,
                    self.dWz, self.dWi, self.dWo, self.dWf,
                    self.Rz, self.Ri, self.Ro, self.Rf,
                    self.dRz, self.dRi, self.dRo, self.dRf,
                    self.bz, self.bi, self.bo, self.bf,
                    self.dbz, self.dbi, self.dbo, self.dbf]
        state = [1, self.has_forgetgate, self.max_seq_len,
                 self.size, self.n_inputs, self.batch_size,
                 self.inputfunction, self.outputfunction,
                 self.dtype, weightstate]
        return state

    def __setstate__(self, state):
        [fileversion, self.has_forgetgate, self.max_seq_len,
                 self.size, self.n_inputs, self.batch_size,
                 self.inputfunction, self.outputfunction,
                 self.dtype, weightstate] = state
        self.setup() # reset all variables (this needs self.size, ...)

        [self.Wz, self.Wi, self.Wo, self.Wf,
            self.dWz, self.dWi, self.dWo, self.dWf,
            self.Rz, self.Ri, self.Ro, self.Rf,
            self.dRz, self.dRi, self.dRo, self.dRf,
            self.bz, self.bi, self.bo, self.bf,
            self.dbz, self.dbi, self.dbo, self.dbf] = weightstate
        self.set_activation_function(self.inputfunction, self.outputfunction)
