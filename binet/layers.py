# -*- coding: utf-8 -*-
'''
A Deep Neural Net implementation

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''
from __future__ import print_function
from __future__ import division
from __future__ import absolute_import

import numpy as np
from binet import op

class BasicLayer(object):
    '''
    Layer in a Neural Network that can perform dropout on its inputs.

    Note that in most implementations, dropout is performed on the output.
    Here we deviate from that, which has two inplications:
        * it's the next layer's task to perform dropout on our activations
        * we ourselves modify the underlying layer's activations (this assumes
          that we get passed lowerlayer.A as input to self.fprop).

    This has several advantages:
        1. Weight scaling comes naturally
        2. Input-Dropout comes naturally
        3. The output layer does not have to do anything special

    This layer is very unoptimized, meaning it stores many intermediate
    values (e.g. presynaptic activations and dropout masks) that you
    often don't need when using a neural net. But it's useful for some
    debugging purposes. However in most cases you will want to use one
    of the more faster implementations.
    '''

    def __init__(self, size, activation="relu", dropout = 0.0,
                 l2_penalty=0.0, l1_penalty=0.0,
                 is_canonical_top_layer = False,
                 is_input_layer=False, activationparams=None,
                 dtype=np.float32):
        self.size = size
        self.dropout = dropout
        self.dropout_value = 0.0
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.is_canonical_top_layer = is_canonical_top_layer
        self.is_input_layer = is_input_layer
        self.activation = activation.lower()
        self.dtype = dtype
        self.W = None
        self.b = None
        self.dW = None
        self.db = None
        self.Z = None  # Presynaptic activations in last fprop call
        self.A = None  # Output of last fprop call
        self.X = None  # Input to last fprop call
        self.M = None  # Last dropout Mask
        self.activationparams=activationparams
        self.set_activation_function(self.activation, activationparams)

    def setup(self, input_shape, batch_size=64):
        n_inputs = np.prod(input_shape) # in case inputs are multidim (e.g. timeseries)
        n_outputs = self.size
        if self.activation == "relu":
            s = np.sqrt(2 / (n_outputs))  # http://arxiv.org/abs/1502.01852
        else:
            s = np.sqrt(2 / (n_inputs + n_outputs)) # Glorot & Bengio, 2010
        s = max(s, 0.02)  # for some reason, large s don't work very well?!?
        #self.W = op.rand_gaussian(shape=(n_outputs, n_inputs), mu=0.0, sigma=s,
        #                          dtype=self.dtype, use_gpu=False)

        # TODO: 2016-11-29: for some reason we reverted to this init, but idk why???
        s = np.sqrt(6) / np.sqrt(n_inputs)
        self.W = op.rand_uniform((n_outputs, n_inputs), -s, +s)
        self.b = np.zeros((1, n_outputs), dtype=self.dtype)
        self.dW = np.zeros_like(self.W)
        self.db = np.zeros_like(self.b)

    activationfunctions = {
        'sigmoid': (op.sigmoid, op.dsigmoid_delta),
        'relu':    (op.relu, op.drelu_delta),
        'linear':  (op.identity, op.dlinear_delta),
        'tanh':    (op.tanh, op.dtanh_delta),
        'softmax': (op.softmax, op.identity),
        'leakyrelu': (op.leakyrelu, op.dleakyrelu_delta),
        'elu': (op.elu, op.delu_delta)
        }


    def set_activation_function(self, activation, params=None):
        if activation == 'clip' or activation == 'clamp':
            def clip(x, out=None):
                return op.clip(x, params[0], params[1], out=out)
            self.func = clip
            self.dfunc = lambda x : (x > params[0]) * (x < params[1])
        elif activation == 'leakyrelu':
            beta = params if params is not None else 0.1
            def fwdfunc(x, out=None, stream=None):
                return op.leakyrelu(x, beta, out, stream)
            def bwdfunc(D, A, X, stream=None):
                return op.dleakyrelu_delta(D, A, X, beta, stream)
            self.func = fwdfunc
            self.dfunc = bwdfunc
        else:
            self.func, self.dfunc = BasicLayer.activationfunctions[activation]

        if self.activation == "softmax":
            assert self.is_canonical_top_layer  # TODO handle non-canonical case

        # if the activation function is the canonical link to the
        # activation function we are using, self.dfunc doesn't appear
        # int he backprop formulation
        if self.is_canonical_top_layer:
            self.dfunc = op.identity

        self.activation = activation
        self.activationparams = params

    def _corrupt_input(self, X, stream=None):
        if self.dropout > 0.0:
            return op.randomly_replace_elements(X, self.dropout, self.dropout_value, stream=stream)
        else:
            return X, None

    def fprop(self, X, stream=None):
        ''' Forward propagation.
        NOTE: If we do dropout, X will get mutated. Usually, X == lowerlayer.A.
              Thus, we don't need to multiply by a dropout mask in bprop.'''
        self.X, self.M = self._corrupt_input(X, stream=stream)
        self.Z = op.dot(self.X, self.W, False, True, stream=stream)
        self.Z = op.add_matvec(self.Z, self.b, out=self.Z, stream=stream)
        self.A = self.func(self.Z, stream=stream)
        return self.A

    def bprop(self, delta, momentum=0.0):
        op.streams[2].synchronize()  # make sure layer above is done
        self.dfunc(delta, self.A, self.Z, stream=op.streams[0])
        op.streams[0].synchronize()
        op.add_dot(delta, self.X, self.dW, True, False,
                  alpha=1.0/delta.shape[0], beta=momentum, stream=op.streams[0])
        m = op.mean(delta, axis=0, stream=op.streams[1])
        op.add_vec(self.db, 1.0, m, beta=momentum, stream=op.streams[1])

        if self.l2_penalty > 0:
            op.add_vec(self.dW, self.l2_penalty, self.W, stream=op.streams[0])

        if not self.is_input_layer:
            if self.dropout > 0.0 and self.activation not in ("relu", "sigmoid"):
                return op.dot(delta, self.W)*self.M
            else:
                return op.dot(delta, self.W)
        else:
            return 0.0

    def update(self, learning_rate, stream=None):
        if self.l1_penalty > 0:
            op.add_vec_l1reg(self.W, self.dW, -learning_rate,
                                   self.l1_penalty, out=self.W, stream=stream)
        else:
            op.add_vec(self.W, -learning_rate, self.dW, stream=stream)
        op.add_vec(self.b, -learning_rate, self.db, stream=stream)


    def __getstate__(self):
        state = [2, self.W, self.b, self.dW, self.db, self.activation,
                self.dropout, self.l2_penalty, self.l1_penalty,
                self.is_canonical_top_layer, self.is_input_layer, self.dtype, self.size,
                self.activationparams, self.dropout_value]
        return state

    def __setstate__(self, state):
        fileversion, self.W, self.b, self.dW, self.db, self.activation, \
            self.dropout, self.l2_penalty, self.l1_penalty, \
            self.is_canonical_top_layer, self.is_input_layer, self.dtype = state[0:12]
        if fileversion > 1:
            self.size = state[12]
            self.activationparams = state[13]
            self.dropout_value = state[14]
        else:
            self.size = self.W.shape[0]
            self.activationparams = None
            self.dropout_value = 0.0


        self.set_activation_function(self.activation, self.activationparams)


class FastDropoutLayer(BasicLayer):
    '''
    Note: This implementation  make use of the fact that we don't backpropagate
    stuff when the activations are zero.
    This is the case for ReLU and sigmoid, but not for tanh. Thus it's
    better to use the DropinLayer for tanh if you want dropout!
    '''
    def __init__(self, *args, **kwargs):
        super(FastDropoutLayer, self).__init__(*args, **kwargs)

    def fprop(self, X, stream=None):
        ''' Forward propagation.
        NOTE: If we do dropout, X will get mutated. Usually, X == lowerlayer.A.
              Thus, we don't need to multiply by a dropout mask in bprop.'''
        self.X, self.M = self._corrupt_input(X, stream=stream)
        self.Z = op.dot(self.X, self.W, False, True, stream=stream)
        self.Z = op.add_matvec(self.Z, self.b, out=self.Z, stream=stream)
        self.A = self.func(self.Z, stream=stream)
        return self.A

    def bprop(self, delta, momentum=0.0):
        op.streams[2].synchronize()  # make sure layer above is done
        self.dfunc(delta, self.A, self.Z, stream=op.streams[0])
        op.streams[0].synchronize()
        op.add_dot(delta, self.X, self.dW, True, False,
                  alpha=1.0/delta.shape[0], beta=momentum, stream=op.streams[0])
        m = op.mean(delta, axis=0, stream=op.streams[1])
        op.add_vec(self.db, 1.0, m, beta=momentum, stream=op.streams[1])

        if self.l2_penalty > 0:
            op.add_vec(self.dW, self.l2_penalty, self.W, stream=op.streams[0])

        if not self.is_input_layer:
            return op.dot(delta, self.W, stream=op.streams[2])
        else:
            return 0.0


def inplace_dropin(X, rate, value, stream=None):
    if isinstance(X, op.gpuarray.GPUArray):
        return op.randomly_replace_elements(X, rate, value, stream=stream)
    else:
        M = (op.rand_uniform_like(X) < rate)
        X[M.astype(np.bool)] = value
    return X, M


class DropinLayer(BasicLayer):
    ''' Does Dropin instead of Dropout.

    Note: re-uses self.dropout as dropin_rate, just so the 'skip dropout'
    logic does not need to be changed.
    '''
    def __init__(self, *args, **kwargs):
        super(DropinLayer, self).__init__(*args, **kwargs)

        if 'activationparams' in kwargs and kwargs['activationparams'] is not None:
            self.dropin_value = kwargs['activationparams'][1]
        else:
            self.dropin_value = 5.0

    def set_activation_function(self, activation, params=None):
        if activation == 'clip':
            def clip(x, out=None):
                return op.clip(x, params[0], params[1], out=out)
            self.func = clip
            self.dfunc = lambda x : (x > params[0]) * (x < params[1])
        else:
            self.func, self.dfunc = BasicLayer.activationfunctions[activation]

    def _corrupt_input(self, X, stream=None):
        if self.dropout > 0.0:
            return inplace_dropin(X, self.dropout, self.dropin_value, stream=stream)
        else:
            return X, None

    def __getstate__(self):
        state = super(DropinLayer, self).__getstate__()
        state.append(self.dropin_value)
        return state

    def __setstate__(self, state):
        self.dropin_value = state.pop()
        super(DropinLayer, self).__setstate__(state)


try:
    from pycuda.elementwise import ElementwiseKernel

    # the mask says wether we change the element, d says if we set to 0 or to 1
    __cuda_inplace_saltpepper = ElementwiseKernel(
        'float* x, float* m, float* d, float overall_rate, float salt_rate, float salt_value',
        """
        m[i] = m[i] > overall_rate;
        x[i] = m[i] ? x[i] : (d[i] < salt_rate ? salt_value : 0.0f);
        """,
        'eltw_saltpepper_inplace')
except ImportError:
    pass

def inplace_saltpepper(X, overall_rate, salt_rate, salt_value, stream=None):
    M = op.rand_uniform_like(X, stream=op.streams[0])
    D = op.rand_uniform_like(X, stream=op.streams[1])
    if isinstance(X, op.gpuarray.GPUArray):
        __cuda_inplace_saltpepper(X, M, D, overall_rate, salt_rate,
                                  salt_value, stream=stream)
    else:
        M = M > overall_rate
        D = D < salt_rate
        D = D.astype(X.dtype)
        if salt_value != 1.0:
            D *= salt_value
        X[~M.astype(np.bool)] = D[~M.astype(np.bool)]
    return X, M


class SaltPepperLayer(DropinLayer):
    def __init__(self, *args, **kwargs):
        super(SaltPepperLayer, self).__init__(*args, **kwargs)
        self.salt_value = 1.0
        self.salt_rate = 0.5

    def _corrupt_input(self, X, stream=None):
        if self.dropout > 0.0:
            return inplace_saltpepper(X, self.dropout, self.salt_rate,
                                      self.salt_value, stream=stream)
        else:
            return X, None

    def __getstate__(self):
        state = super(SaltPepperLayer, self).__getstate__()
        state.append(self.salt_value)
        state.append(self.salt_rate)
        return state

    def __setstate__(self, state):
        self.salt_rate = state.pop()
        self.salt_value = state.pop()
        super(SaltPepperLayer, self).__setstate__(state)


# TODO: find a nice way to integrate AdaGrad into everything!
class AdaGradLayer(FastDropoutLayer):
    def __init__(self, *args, **kwargs):
        super(FastDropoutLayer, self).__init__(*args, **kwargs)

    def setup(self, input_shape, batch_size=64):
        super(FastDropoutLayer, self).setup(input_shape, batch_size)
        self.gW = np.ones_like(self.dW)
        self.gb = np.ones_like(self.db)


    def update(self, learning_rate, stream=None):
        self.gW += self.dW*self.dW
        self.gb += self.db*self.db
        self.W -= (learning_rate / op.sqrt(self.gW))*self.dW
        self.b -= (learning_rate / op.sqrt(self.gb))*self.db
