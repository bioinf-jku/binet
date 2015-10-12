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
from binet import NeuralNet, load_dataset

op.init_gpu()

X, y, Xval, yval = load_dataset("mnist_basic")
Xd, yd = op.to_gpu(X.copy()),op.to_gpu(y.copy())

def test_gpucpu_fprop_equality():
    '''Test forward propagation CPU/GPU equality.'''
    neth = NeuralNet([X.shape[1], 128, 32, y.shape[1]])
    netd = op.to_gpu(copy.deepcopy(neth))
    outh = neth.forward_pass(X)
    outd = netd.forward_pass(Xd)
    assert_allclose(outd.get(), outh, rtol=1e-5, err_msg="frop error")


def test_gpucpu_bprop_equality():
    '''Test backpropagation CPU/GPU equality.'''

    neth = NeuralNet([X.shape[1], 128, 32, y.shape[1]])
    netd = op.to_gpu(copy.deepcopy(neth))

    outh = neth.forward_pass(X)
    lh = neth.backward_pass(outh, y)

    outd = netd.forward_pass(Xd)
    ld = netd.backward_pass(outd, yd)

    assert_almost_equal(lh, ld)
    assert_allclose(outd.get(), outh, rtol=1e-5, err_msg="frop error")
    for i in reversed(range(len(neth.layersizes)-1)):
        dwh = neth.layers[i].dW
        dbh = neth.layers[i].db
        dwd = netd.layers[i].dW.get()
        dbd = netd.layers[i].db.get()
        assert_allclose(dwh, dwd, atol=1e-5, err_msg="dW diff in layer %d" % i)
        assert_allclose(dbh, dbd, atol=1e-5, err_msg="db diff in layer %d" % i)
