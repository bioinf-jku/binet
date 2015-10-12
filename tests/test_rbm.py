import sys
import re

import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, lil_matrix
from sklearn.utils.testing import (assert_almost_equal, assert_array_equal,
                                   assert_true)

from sklearn.datasets import load_digits
from sklearn.externals.six.moves import cStringIO as StringIO
from sklearn.utils.validation import assert_all_finite

from binet.rbm import BernoulliRBM

np.seterr(all='warn')

Xdigits = load_digits().data
Xdigits -= Xdigits.min()
Xdigits /= Xdigits.max()


def test_nomodification():
    X = Xdigits.copy().astype(np.float32)

    rbm = BernoulliRBM(X.shape[1], n_hidden=64, learning_rate=0.1,
                       batch_size=10, n_iter=7, random_state=9)
    rbm.fit(X)

    # in-place tricks shouldn't have modified X
    assert_array_equal(X, Xdigits)


def test_partial_fit():
    X = Xdigits.copy().astype(np.float32)
    rbm = BernoulliRBM(X.shape[1], n_hidden=64, learning_rate=0.1,
                       batch_size=20, random_state=9)
    n_samples = X.shape[0]
    n_batches = int(np.ceil(float(n_samples) / rbm.batch_size))
    batch_slices = np.array_split(X, n_batches)

    for i in range(7):
        for batch in batch_slices:
            rbm.partial_fit(batch)
    assert_array_equal(X, Xdigits)


def test_transform():
    X = Xdigits[:100].astype(np.float32)
    rbm1 = BernoulliRBM(X.shape[1], n_hidden=16, batch_size=5,
                        n_iter=5, random_state=42)
    rbm1.fit(X)

    Xt1 = rbm1.transform(X)
    Xt2 = rbm1._mean_hiddens(X)

    assert_array_equal(Xt1, Xt2)


def test_sample_hiddens():
    rng = np.random.RandomState(0)
    X = Xdigits[:100].astype(np.float32)
    rbm1 = BernoulliRBM(X.shape[1], n_hidden=2, batch_size=5,
                        n_iter=5, random_state=42)
    rbm1.fit(X)

    h = rbm1._mean_hiddens(X[0])
    hs = np.mean([rbm1._sample_hiddens(X[0]) for i in range(100)], 0)

    assert_almost_equal(h, hs, decimal=1)


def test_gibbs_smoke():
    """Check if we don't get NaNs sampling the full digits dataset."""
    rng = np.random.RandomState(42)
    X = Xdigits.astype(np.float32)
    rbm1 = BernoulliRBM(X.shape[1], n_hidden=42, batch_size=40,
                        n_iter=20, random_state=rng)
    rbm1.fit(X)
    X_sampled = rbm1.gibbs(X)
    assert_all_finite(X_sampled)


def test_rbm_verbose():
    rbm = BernoulliRBM(Xdigits.shape[1], n_iter=2, verbose=10)
    old_stdout = sys.stdout
    sys.stdout = StringIO()
    try:
        rbm.fit(Xdigits.astype(np.float32))
    finally:
        sys.stdout = old_stdout
