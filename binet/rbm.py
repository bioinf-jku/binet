# -*- coding: utf-8 -*-
'''
Restricted Bolzmann Machine, based on sklearn

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)

Note: this code is based on the BernoulliRBM from sklearn 0.15.2
© 2010 - 2014, scikit-learn developers
(BSD License, see http://scikit-learn.org for more information)

# Main sklearn author: Yann N. Dauphin <dauphiya@iro.umontreal.ca>
# sklearn Author: Vlad Niculae
# sklearn Author: Gabriel Synnaeve
'''

import time

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin
from sklearn.externals.six.moves import xrange
from binet import op
from binet.util import generate_slices


class _BaseRBM(BaseEstimator, TransformerMixin):
    """Basic RBM implementation, with unspecified visual distribution.

    Has PCD and CD-1 training options. """
    def __init__(self, n_visibles, n_hidden=256, n_iter=100,
                 learning_rate=0.1, momentum=0.0, batch_size=64,
                 verbose=0, use_pcd=True, random_state=None, dtype=np.float32):
        self.n_visibles = n_visibles
        self.n_hidden = n_hidden
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.batch_size = batch_size
        self.n_iter = n_iter
        self.use_pcd = use_pcd
        self.verbose = verbose
        self.random_state = random_state
        self.dtype = dtype

        ws = (n_hidden, n_visibles)
        self.W = op.rand_gaussian(ws, 0.0, 0.01, dtype)
        self.W = self.W.astype(np.float32)
        self.bh = np.zeros((1, self.n_hidden), dtype=dtype)
        self.bv = np.zeros((1, n_visibles), dtype=dtype)
        self.h_samples_ = np.zeros((self.batch_size, self.n_hidden), dtype=dtype)
        self.dW = np.zeros_like(self.W)
        self.dbh = np.zeros_like(self.bh)
        self.dbv = np.zeros_like(self.bv)

    def transform(self, X):
        """Compute the hidden layer activation probabilities, P(h=1|v=X).

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            The data to be transformed.

        Returns
        -------
        h : array, shape (n_samples, n_components)
            Latent representations of the data.
        """
        return self._mean_hiddens(X)

    def _mean_hiddens(self, v):
        """Computes the probabilities P(h=1|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Corresponding mean field values for the hidden layer.
        """
        p = op.dot(v, self.W, False, True)
        p = op.add_matvec(p, self.bh, out=p)
        return op.sigmoid(p, out=p)

    def _mean_visibles(self, h):
        raise NotImplementedError("Not implemented yet")

    def _sample_hiddens(self, v):
        """Sample from the distribution P(h|v).

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to sample from.

        Returns
        -------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer.
        """
        p = self._mean_hiddens(v)
        return op.sample_binomial(p)

    def _sample_visibles(self, h):
        raise NotImplementedError("Not implemented yet")

    def reconstruct_hiddens(self, h):
        return self._mean_visibles(h)

    def gibbs(self, v):
        """Perform one Gibbs sampling step.

        Parameters
        ----------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer to start from.

        Returns
        -------
        v_new : array-like, shape (n_samples, n_features)
            Values of the visible layer after one Gibbs step.
        """
        h_ = self._sample_hiddens(v)
        v_ = self._sample_visibles(h_)

        return v_

    def partial_fit(self, X):
        """
        Fit the model to the data X which should contain a partial
        segment of the data.

        Adjust the parameters to maximize the likelihood of v using
        Stochastic Maximum Likelihood (SML).

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data to use for training.
        """
        v_pos = X
        h_pos = self._mean_hiddens(v_pos)

        if self.use_pcd:
            v_neg = self._sample_visibles(self.h_samples_)
        else:
            v_neg = self._sample_visibles(h_pos)
        h_neg = self._mean_hiddens(v_neg)

        lr = float(self.learning_rate) / v_pos.shape[0]
        op.add_dot(h_pos, v_pos, self.dW, True, False, alpha=1.0, beta=self.momentum)
        op.add_dot(h_neg, v_neg, self.dW, True, False, alpha=-1.0, beta=1.0)
        self.W += lr * self.dW

        self.dbh *= self.momentum
        self.dbv *= self.momentum
        self.dbh += (op.sum(h_pos, axis=0) - op.sum(h_neg, axis=0)).reshape(1, self.dbh.shape[1])
        self.dbv += (op.sum(v_pos, axis=0) - op.sum(v_neg, axis=0)).reshape(1, self.dbv.shape[1])
        self.bh += lr * self.dbh
        self.bv += lr * self.dbv
        if self.use_pcd:
            self.h_samples_ = op.sample_binomial(h_neg)

    def calculate_reconstruction_rmse(self, X):
        H = self.transform(X)
        R = self._mean_visibles(H)
        d = (op.sum((R-X)**2))/X.shape[0]
        return np.sqrt(op.to_cpu(d))

    def fit(self, X, y=None):
        """Fit the model to the data X.

        Parameters
        ----------
        X : {array-like, sparse matrix} shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : BernoulliRBM
            The fitted model.
        """
        self.h_samples_ *= 0
        begin = time.time()
        for self.current_epoch in xrange(self.n_iter):
            for batch_slice in generate_slices(X.shape[0], self.batch_size):
                self.partial_fit(X[batch_slice])

            if self.verbose:
                end = time.time()
                rmse = self.calculate_reconstruction_rmse(X)
                print("[%s] Iteration %d, ReconstructionRMSE %.4f  time = %.2fs"
                      % (type(self).__name__, self.current_epoch, rmse, end - begin))
                begin = end

        return self


class BernoulliRBM(_BaseRBM):
    """Bernoulli Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with binary visible units and
    binary hiddens. Parameters are estimated using Stochastic Maximum
    Likelihood (SML), also known as Persistent Contrastive Divergence (PCD)
    [2].

    The time complexity of this implementation is ``O(d ** 2)`` assuming
    d ~ n_features ~ n_components.

    Parameters
    ----------
    n_components : int, optional
        Number of binary hidden units.

    learning_rate : float, optional
        The learning rate for weight updates. It is *highly* recommended
        to tune this hyper-parameter. Reasonable values are in the
        10**[0., -3.] range.

    batch_size : int, optional
        Number of examples per minibatch.

    n_iter : int, optional
        Number of iterations/sweeps over the training dataset to perform
        during training.

    verbose : int, optional
        The verbosity level. The default, zero, means silent mode.

    random_state : integer or numpy.RandomState, optional
        A random number generator instance to define the state of the
        random permutations generator. If an integer is given, it fixes the
        seed. Defaults to the global numpy random number generator.

    Attributes
    ----------
    bh : array-like, shape (n_components,)
        Biases of the hidden units.

    bv : array-like, shape (n_features,)
        Biases of the visible units.

    W : array-like, shape (n_components, n_features)
        Weight matrix, where n_features in the number of
        visible units and n_components is the number of hidden units.

    Examples
    --------

    >>> import numpy as np
    >>> from sklearn.neural_network import BernoulliRBM
    >>> X = np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]])
    >>> model = BernoulliRBM(n_components=2)
    >>> model.fit(X)
    BernoulliRBM(batch_size=10, learning_rate=0.1, n_components=2, n_iter=10,
           random_state=None, verbose=0)

    References
    ----------

    [1] Hinton, G. E., Osindero, S. and Teh, Y. A fast learning algorithm for
        deep belief nets. Neural Computation 18, pp 1527-1554.
        http://www.cs.toronto.edu/~hinton/absps/fastnc.pdf

    [2] Tieleman, T. Training Restricted Boltzmann Machines using
        Approximations to the Likelihood Gradient. International Conference
        on Machine Learning (ICML) 2008
    """
    def __init__(self, n_visibles, n_hidden=256, n_iter=10,
                 learning_rate=0.1, momentum=0.0, batch_size=64,
                 verbose=0, use_pcd=True, random_state=None, dtype=np.float32):
        super(BernoulliRBM, self).__init__(n_visibles=n_visibles,
            n_hidden=n_hidden, n_iter=n_iter, learning_rate=learning_rate,
            momentum=momentum, batch_size=batch_size, verbose=verbose,
            use_pcd=use_pcd, random_state=random_state, dtype=dtype)

    def _mean_visibles(self, h):
        """Computes the probabilities P(v=1|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = op.dot(h, self.W, False, False)
        p = op.add_matvec(p, self.bv, out=p)
        return op.sigmoid(p, out=p)

    def _sample_visibles(self, h):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = self._mean_visibles(h)
        return op.sample_binomial(p)


class GaussianRBM(_BaseRBM):
    """Gaussian Restricted Boltzmann Machine (RBM).

    A Restricted Boltzmann Machine with normal distributed visible units and
    binary hiddens.
    """
    def __init__(self, n_visibles, n_hidden=256, n_iter=10,
                 learning_rate=0.1, momentum=0.0, batch_size=64,
                 verbose=0, use_pcd=True, random_state=None, dtype=np.float32):
        super(GaussianRBM, self).__init__(n_visibles=n_visibles,
            n_hidden=n_hidden, n_iter=n_iter, learning_rate=learning_rate,
            momentum=momentum, batch_size=batch_size, verbose=verbose,
            use_pcd=use_pcd, random_state=random_state, dtype=dtype)

    def _mean_visibles(self, h):
        """Computes the probabilities P(v=1|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = op.dot(h, self.W, False, False)
        p = op.add_matvec(p, self.bv, out=p)
        return p

    def _sample_visibles(self, h):
        """Sample from the distribution P(v|h).

        Parameters
        ----------
        h : array-like, shape (n_samples, n_components)
            Values of the hidden layer to sample from.

        Returns
        -------
        v : array-like, shape (n_samples, n_features)
            Values of the visible layer.
        """
        p = self._mean_visibles(h)
        return p + op.rand_gaussian_like(p)
