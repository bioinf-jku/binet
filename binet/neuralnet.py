# -*- coding: utf-8 -*-
'''
A Deep Neural Net implementation

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import numpy as np
import pandas as pd
import os
import time
import sys
import copy
import warnings
from scipy import sparse

from sklearn.preprocessing import OneHotEncoder
from sklearn.utils import check_random_state
from sklearn.base import BaseEstimator
from sklearn.cross_validation import train_test_split

from binet import op
from binet.layers import (BasicLayer, FastDropoutLayer,
                          DropinLayer, SaltPepperLayer, AdaGradLayer)
from binet.util import generate_slices


class NeuralNet(BaseEstimator):
    def __init__(self, n_inputs, layersizes=None, max_iter=100,
                 learning_rate=0.05,
                 activation="ReLU", output="softmax", loss="crossentropy",
                 l2_penalty=0.0, l1_penalty=0.0,
                 dropout=0.0, input_dropout=0.0, batch_size=64,
                 momentum=0.0, fraction_validation_set=0.15,
                 convergence_iter_tol=30,  early_stopping=True,
                 shuffle_data=True,
                 learning_rate_schedule="constant", learning_rate_decay=None,
                 verbose=False, random_state=None, dtype=np.float32,
                 activationparams=None,
                 layerclass="default", logger=None,
                 output_weights=None):
        self.batch_size = batch_size
        self.max_iter = max_iter
        # we need a seed that is random even in multiple threads
        # started very close to each other in time
        if random_state is None:
            random_state = op.rand_int()

        self.n_inputs = n_inputs
        self.dropout = dropout
        self.verbose = verbose
        self.random_state = random_state
        self.l2_penalty = l2_penalty
        self.l1_penalty = l1_penalty
        self.shuffle_data = shuffle_data
        self.learning_rate = learning_rate # used for adaptive schedule
        self.momentum = momentum
        self.learning_rate_schedule = learning_rate_schedule
        self.learning_rate_decay = learning_rate_decay
        self.layersizes = layersizes
        self.input_dropout = input_dropout
        self.activation = activation.lower()
        self.loss = loss.lower()
        self.output = output.lower()
        self.dtype = dtype
        self.layerclass = layerclass
        self.logger = logger
        self.output_weights = output_weights
        self.convergence_iter_tol = convergence_iter_tol
        self.fraction_validation_set = fraction_validation_set
        self.early_stopping = early_stopping
        self.activationparams = activationparams
        if learning_rate_schedule not in ('constant', 'adaptive', 'simple', 'invscale', 'linear', 'power'):
            raise ValueError("Unknown learning rate schedule.")
        self.layers = []
        if self.layersizes is not None:
            self.setup_layers(self.activationparams)

        if self.random_state is not None:
            op.set_seed(self.random_state)
        self.reset()

    layerclasses = {'default': FastDropoutLayer, 'basic': BasicLayer,
                    'dropin': DropinLayer, 'saltpepper': SaltPepperLayer,
                    'adagrad': AdaGradLayer}

    @classmethod
    def getLayerClass(cls, name):
        return NeuralNet.layerclasses[name]

    def setup_layers(self, activationparams=None):
        LayerClass = NeuralNet.getLayerClass(self.layerclass)
        for i, layersize in enumerate(self.layersizes):
            actfun = self.activation
            is_output_canonical = False
            if i == len(self.layersizes) - 1:
                actfun = self.output
                if self.loss == "crossentropy" and (self.output == "softmax" or self.output == "sigmoid"):
                    is_output_canonical = True
                elif self.loss == "squarederror" and (self.output == "linear"):
                    is_output_canonical = True
            self.layers.append(LayerClass(layersize, activation=actfun,
                dropout=self.dropout, l2_penalty=self.l2_penalty,
                l1_penalty=self.l1_penalty, is_input_layer=(i==0),
                is_canonical_top_layer=is_output_canonical,
                activationparams=activationparams, dtype=self.dtype))
        self.layers[0].dropout = self.input_dropout

    def reset(self, random_state=None):
        self.statistics = pd.DataFrame(dtype=np.float64,
            columns=('train_error', 'val_error', 'val_score', 'time'))
        self.statistics.index.name = "epoch"
        self.current_epoch = 0 # number of iterations
        self.update_count = 0
        self._best_params = None
        self._best_score_va = np.finfo(np.float32).min
        self._no_improvement_since = 0
        if random_state is not None:
            op.set_seed(self.random_state)
        if len(self.layers) > 0:
            self.layers[0].setup(self.n_inputs, batch_size=self.batch_size)
        for l1, l2 in zip(self.layers[:-1], self.layers[1:]):
            l2.setup(l1.size, batch_size=self.batch_size)

    def forward_pass(self, X):
        # copy because of input-dropout
        if self.layers[0].dropout > 0.0:
            a = op.copy(X, stream=op.streams[0])
        else:
            a = X
        for l in self.layers:
            a = l.fprop(a, stream=op.streams[0])
        return a

    def backward_pass(self, out, y, momentum=0.0):
        delta = op.toplayer_delta(out, y, self.layers[-1].Z,
                                  stream=op.streams[0])
        if self.output_weights is not None:
            op.inplace_scale_columns(delta, self.output_weights)
        for l in reversed(self.layers):
            delta = l.bprop(delta, momentum)
        return 0.0

    def _get_current_learningrate(self, t):
        ''' Gets the learningrate and momentum parameters for epoch t.'''
        if isinstance(self.learning_rate, (list, tuple)):
            lr = self.learning_rate
        else:
            lr = [self.learning_rate] * len(self.layers)
        lr = np.array(lr)  # there would be trouble with op.to_op(net)
                           # if we converted this sooner!
        mu = self.momentum

        if self.learning_rate_schedule == 'adaptive':
            mu = 1- 2 ** (-1 -np.log(np.floor(t/250.0) +1))
            lr = 1 /(t + (1/lr))
        elif self.learning_rate_schedule == "invscale":
            f = 0.5 if self.learning_rate_decay is None else self.learning_rate_decay
            lr = lr /(t+1)**(f)
        elif self.learning_rate_schedule == "power":
            f = 2.0 if self.learning_rate_decay is None else self.learning_rate_decay
            lr = lr * (1.0 - ((t/self.max_iter)**f))
        elif self.learning_rate_schedule == "linear":
            lr = (lr - (t * lr) / (self.max_iter + 1))
        elif self.learning_rate_decay is not None:  # 'simple' schedule
            f = 0.99 if self.learning_rate_decay is None else self.learning_rate_decay
            lr = f**t * lr
        else:
            lr = lr
        return (lr, mu)

    def partial_fit(self, X, y, encode_labels=True):
        ''' Runs one epoch of minibatch-backprop on the given data.

        Note: Input-Dropout might overwrite parts of X!

        Expects y in One-Hot format'''
        assert len(self.layers) <= len(op.streams) # needed when we call l.update
        if not sparse.isspmatrix_csr(X):
            assert(X.flags.c_contiguous)

        cur_lr, cur_momentum = self._get_current_learningrate(self.current_epoch)
        err = 0.0
        for s in generate_slices(X.shape[0], self.batch_size):
            Xtemp = X[s]
            ytemp = y[s]

            # for sparse matrices, the fastest option is to convert to
            # dense on the GPU and then operate in dense
            if sparse.isspmatrix_csr(X) and isinstance(self.layers[0].W, op.gpuarray.GPUArray):
                a = op.cuda_memory_pool.allocate
                #Xtemp = op.to_gpu(Xtemp.A, stream=op.streams[0])
                #ytemp = op.to_gpu(ytemp, stream=op.streams[1])
                Xtemp = op.GPUCSRArray(Xtemp, allocator=a, stream=op.streams[0])
                Xtemp = Xtemp.todense(allocator=a, stream=op.streams[0])
                if sparse.isspmatrix_csr(ytemp):
                    ytemp = op.to_gpu(ytemp.toarray(), stream=op.streams[1])
                else:
                    ytemp = op.to_gpu(ytemp, stream=op.streams[1])

            out = self.forward_pass(Xtemp)
            op.streams[1].synchronize()
            self.backward_pass(out, ytemp, cur_momentum)
            op.streams[2].synchronize()
            for i, l in enumerate(self.layers):
                l.update(cur_lr[i], stream=op.streams[i])
            self.update_count += 1

            err += self._get_loss(ytemp, out)
        self.current_epoch += 1
        return err / y.shape[0]

    def fit(self, X, y, X_va=None, y_va=None, skip_output=-1):
        y, y_va = self._check_y_shape(y, y_va)
        assert X.shape[0] == y.shape[0], "X and y have the a different number of samples"
        assert X.shape[1] == self.n_inputs, "X doesn't have the right number of features"
        if X_va is not None:
            assert X_va.shape[0] == y_va.shape[0], "X_va and y_va have the a different number of samples"
            assert X_va.shape[1] == X.shape[1], "X_va doesn't have the right number of features"


        # PyCUDA, doesn't allow shuffling via indexing (`X[idx]`) so to have
        # a random split, we just shuffle the data (if we're allowed to)
        if X_va is None and self.fraction_validation_set > 0.0:
            if self.shuffle_data:
                idx = np.arange(X.shape[0])
                Xn, yn = op.shuffle_rows(X, y, idx=idx)
            else:
                warnings.warn("using first part of X as validation set without shuffling first")
            vi = int(y.shape[0]*self.fraction_validation_set)
            X, X_va, y, y_va = X[vi:], X[:vi], y[vi:], y[:vi]

        oldverbose = self.verbose
        try:
            # generate storage for shuffling now
            if self.shuffle_data:
                idx = np.arange(X.shape[0])
                Xn, yn = op.shuffle_rows(X, y, idx=idx)
            t0 = time.time()
            for i in range(self.current_epoch, self.max_iter):
                if self.shuffle_data:
                    Xn, yn = op.shuffle_rows(X, y, output=(Xn, yn), idx=idx)
                else:
                    Xn, yn = X, y
                err = float(self.partial_fit(Xn, yn, encode_labels=False))
                if oldverbose and skip_output > 0:
                    self.verbose = (self.current_epoch % skip_output == 0)
                self.track_progress(t0, err, Xn, yn, X_va, y_va)
                if self._no_improvement_since >= self.convergence_iter_tol and self.early_stopping:
                    for i, l in enumerate(self.layers):
                        l.W = self._best_params[0][i]
                        l.b = self._best_params[1][i]
                    self.current_epoch -= self._no_improvement_since
                    break
        finally:
            self.verbose = oldverbose

    def track_progress(self, t0, error_tr, X, y, X_va=None, y_va=None):
        y, y_va = self._check_y_shape(y, y_va)
        dt = time.time() - t0
        if X_va is not None:
            out = self.transform(X_va)
            error_va = self._get_loss(y_va, out)
            score_va = self._get_score(y_va, out)

            if self.early_stopping and score_va > self._best_score_va and self.current_epoch > 0:
                self._best_params = (copy.deepcopy(self.weights), copy.deepcopy(self.bias))
                self._best_score_va = score_va
                self._no_improvement_since = 0
            else:
                self._no_improvement_since += 1
        else:
            score_va, error_va = -1, -1
        if self.verbose:
            vstr = "Val-Loss: %3.6f\tVal-Score: %5.4f%%\t" % (error_va, score_va*100.0)
            msg = "%3d:\tTrain-Loss: %3.6f\t%s(%3.2fs)" % (self.current_epoch, error_tr, vstr, dt)
            if self.logger is None:
                print(msg)
                sys.stdout.flush()
            else:
                self.logger.info(msg)
        #self.statistics.append((error_tr, verr, score_va, dt, self.current_epoch)
        self.statistics.loc[self.current_epoch] = (error_tr, error_va, score_va, dt)

    def _get_loss(self, target, pred):
        if self.loss == "crossentropy":
            op.streams[0].synchronize()
            return op.cross_entropy(target, pred, stream=op.streams[3])
        elif self.loss == "squarederror":
            return op.mean_squared_error(target, pred)
        else:
            raise NotImplementedError()

    def _get_score(self, target, pred):
        '''Calculates the quality of predictions of a model.

        Like sklearn, we follow the convention that higher return values are
        better than lower return values.'''

        if self.output == "softmax":
            # convert from 1hot
            if len(target.shape) != 1 and target.shape[1] != 1:
                target = op.to_cpu(op.argmax(target, 1))
            if len(pred.shape) != 1 and pred.shape[1] != 1:
                pred = op.to_cpu(op.argmax(pred, 1))
            acc = op.to_cpu(op.mean(pred == target))
            return float(acc)
        elif self.output == "sigmoid":
            # Note: this is meant for multitask learning, but for e.g.
            # using sigmoid+squarederror as multiclass problem, this will
            # give the wrong result!
            return op.to_cpu(op.mean(target == (pred > 0.5)))
        elif self.output == "linear":
            return -op.to_cpu(op.mean((target - pred)**2)) # negate by convention
        else:
            raise NotImplementedError()

    def score(self, X, y):
        p = self.transform(X)
        y, _ = self._check_y_shape(y, None)
        return self._get_score(y, p)

    def transform(self, X):
        '''Transforms the input X into predictions.
        Note: this essentially runs the forward pass, but without using dropout.
        '''
        # We run in batch mode so we're sure not to use more memory than
        # the training forward passes.
        out = op.empty((X.shape[0], self.layers[-1].size),
                       dtype=self.dtype,
                       use_gpu=type(X) == op.gpuarray.GPUArray)
        for s in generate_slices(X.shape[0], self.batch_size):
            a = X[s]
            for i, l in enumerate(self.layers):
                odr = 0.0
                if l.dropout > 0:
                    #if i > 0:  # dont scale in the input layer
                    l.W *= (1.0 - l.dropout)
                    odr, l.dropout = l.dropout, 0.0
                a = l.fprop(a, stream=op.streams[0])
                if odr > 0:
                    l.dropout = odr
                    #if i > 0:
                    l.W /= (1.0 - l.dropout)
            out[s] = a
        return out

    def predict(self, X):
        out = self.transform(X)
        if self.output == "softmax":
            return op.argmax(out, 1)
        elif self.output == "sigmoid":
            return out > 0.5
        else:
            return out

    def predict_proba(self, X):
        p = self.transform(X)
        if self.output == "sigmoid":
            return np.hstack([1.0-p, p])
        else:
            return p

    def _check_y_shape(self, y, y_va):
        if self.output == "softmax" and (len(y.shape) == 1 or y.shape[1] == 1):
            enc = OneHotEncoder(dtype=self.dtype)
            y = enc.fit_transform(y).toarray()
            if y_va is not None:
               y_va = enc.transform(y_va).toarray()
        elif len(y.shape) == 1:
            y = y.reshape(-1, 1)
            if y_va is not None:
               y_va = y_va.reshape(-1, 1)
        if y.dtype != self.dtype:
            y = y.astype(self.dtype)
            if y_va is not None:
               y_va = y_va.astype(self.dtype)
        return y, y_va

    def __getstate__(self):
        state = [14, self.batch_size, self.max_iter, self.learning_rate, self.dropout,
            self.input_dropout, self.verbose, self.random_state, self.layers,
            self.current_epoch, self.momentum,
            self.activation, self.statistics, self.layersizes, self.dtype,
            self.learning_rate_schedule, self.learning_rate_decay, self.shuffle_data,
            self.output, self.layerclass, self.l2_penalty, self.l1_penalty,
            self.loss, self.output_weights, self.update_count,
            self.convergence_iter_tol, self.fraction_validation_set,
            self.early_stopping, self.activationparams, self.n_inputs]
        return state

    def __setstate__(self, state):
        fileversion = state[0]
        if fileversion < 12:
            # we removed the different learning_rate schedule variables from
            # the middle. Use commits from before 2015-04-02
            raise RuntimeError("Unsupported file version")

        self.batch_size, self.max_iter, self.learning_rate, self.dropout, \
            self.input_dropout, self.verbose, self.random_state, self.layers, \
            self.current_epoch, self.momentum, \
            self.activation, self.statistics, self.layersizes, self.dtype, \
            self.learning_rate_schedule, self.learning_rate_decay, self.shuffle_data, \
            self.output, self.layerclass, self.l2_penalty, self.l1_penalty, \
            self.loss, self.output_weights, self.update_count, \
            self.convergence_iter_tol, self.fraction_validation_set, \
            self.early_stopping, self.activationparams, \
            self.n_inputs = state[1:30]
        self._no_improvement_since = 0

    @property
    def weights(self):
        return [l.W for l in self.layers]

    @property
    def bias(self):
        return [l.b for l in self.layers]

    def set_params(self, **params):
        '''Set the parameters of the neural net.

        This method is called e.g. by sklearn's GridSearchCV when
        trying out different parameter settings. We need to reset the net to
        recreate the layers with the new parameters afterwards.'''
        super(NeuralNet, self).set_params(**params)
        if self.layersizes is not None:
            self.setup_layers(self.activationparams)
        self.reset(self.random_state)
        return self
