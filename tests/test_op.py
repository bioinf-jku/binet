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
op.init_gpu()


def test_togpu():
    X = np.random.randn(3, 5)
    Xd = op.to_gpu(X)
    assert type(Xd) == gpuarray.GPUArray
    assert Xd.shape == X.shape

    Xd2 = op.to_gpu(Xd)
    assert Xd2 is Xd


def test_tonumpy():
    X = np.random.randn(3, 5)
    Xd = op.to_gpu(X)
    Xh = op.to_cpu(Xd)
    assert type(Xh) == np.ndarray
    assert Xh.shape == X.shape
    assert_allclose(Xh, X)

    X2 = op.to_cpu(X)
    assert X2 is X


def test_togpu_class():
    class MyTest:
        def __init__(self):
            self.X = np.random.randn(3, 5)
    t = MyTest()
    Td = op.to_gpu(t)
    assert type(Td.X) == gpuarray.GPUArray, "type is %s" % type(Td.X)
    assert Td.X.shape == (3, 5)


def test_tonumpy_class():
    class MyTest:
        def __init__(self):
            self.X = np.random.randn(3, 5)
    t = MyTest()
    Td = op.to_gpu(t)
    Th = op.to_cpu(Td)
    assert type(Th.X) == np.ndarray
    assert Th.X.shape == (3, 5)


def test_tognumpy_list():
    X = [np.random.randn(3, 5), "teststring"]
    Xd = op.to_gpu(X)
    Xh = op.to_cpu(Xd)
    assert type(Xh[0]) == np.ndarray
    assert Xh[0].shape == X[0].shape
    assert_array_equal(Xh[0], X[0])

def test_togpu_list():
    X = [np.random.randn(3, 5), "teststring"]
    X_orig = copy.deepcopy(X)
    Xd = op.to_gpu(X)
    assert type(Xd[0]) == op.gpuarray.GPUArray
    assert Xd[0].shape == X_orig[0].shape
    Xh = op.to_cpu(Xd[0])
    assert_allclose(Xh, X_orig[0])


def test_togpu_dict():
    X = {'arr': np.random.randn(3, 5), 'str': "teststring"}
    X_orig = copy.deepcopy(X)
    Xd = op.to_gpu(X)
    assert type(Xd['arr']) == op.gpuarray.GPUArray
    assert Xd['arr'].shape == X_orig['arr'].shape
    Xh = op.to_cpu(Xd['arr'])
    assert_allclose(Xh, X_orig['arr'])


def run_function(X, Y_expected, func, rtol=1e-6, with_inplace_test=True, **kwargs):
    # CPU, with target argument
    Y = np.empty_like(Y_expected)
    Yhr = func(X, out=Y, **kwargs)
    assert_allclose(Y_expected, Yhr, err_msg="CPU with target", rtol=rtol)
    assert Yhr is Y

    # CPU, no target argument
    Yhr = func(X, **kwargs)
    assert_allclose(Y_expected, Yhr, err_msg="CPU, no target", rtol=rtol)

    if with_inplace_test:
        X2 = X.copy()
        Yhr = func(X2, out=X2, **kwargs)
        assert_allclose(Y_expected, Yhr, err_msg="CPU, inplace target", rtol=rtol)
        assert Yhr is X2

    kwargs = op.to_gpu(kwargs)

    # GPU, with target
    Xd = op.to_gpu(X)
    Yd = gpuarray.empty_like(op.to_gpu(Y_expected))
    Ydr = func(Xd, out=Yd, **kwargs)
    assert_allclose(Y_expected, op.to_cpu(Ydr), err_msg="GPU with target", rtol=rtol)
    assert Ydr is Yd

    # GPU, no target
    Ydr = func(Xd, **kwargs)
    assert_allclose(Y_expected, op.to_cpu(Ydr), err_msg="GPU, no target", rtol=rtol)

    if with_inplace_test:
        Ydr = func(Xd, out=Xd, **kwargs)
        assert_allclose(Y_expected, op.to_cpu(Ydr), err_msg="GPU, inplace target", rtol=rtol)
        assert Ydr is Xd


def run_function_with_axis(X, ax0_expected, ax1_expected, noax_expected, func, rtol=1e-6):
    # CPU, no target argument
    ah0 = func(X, axis=0)
    assert_allclose(ax0_expected, ah0, err_msg="CPU, axis=0", rtol=rtol)
    ah1 = func(X, axis=1)
    assert_allclose(ax1_expected, ah1, err_msg="CPU, axis=1", rtol=rtol)
    if noax_expected is not None:
        ah = func(X)
        assert_allclose(noax_expected, ah, err_msg="CPU, axis=1", rtol=rtol)

    Xd = op.to_gpu(X)
    # GPU, no target
    ad0 = func(Xd, axis=0)
    assert_allclose(ax0_expected, op.to_cpu(ad0), err_msg="GPU, axis=0", rtol=rtol)
    ad1 = func(Xd, axis=1)
    assert_allclose(ax1_expected, op.to_cpu(ad1), err_msg="GPU, axis=1", rtol=rtol)
    if noax_expected is not None:
        ad = func(Xd)
        assert_allclose(noax_expected, op.to_cpu(ad), err_msg="GPU, axis=1", rtol=rtol)


def test_relu():
    X = np.random.randn(3, 5).astype(np.float32)
    Y_expected = X.copy()
    Y_expected[Y_expected <= 0.0] = 0.0
    run_function(X, Y_expected, op.relu)

def test_abs():
    X = np.random.randn(3, 5).astype(np.float32)
    Y_expected = X.copy()
    Y_expected[Y_expected <= 0.0] *= -1
    run_function(X, Y_expected, op.abs)


def test_sigmoid():
    X = np.random.randn(3, 5).astype(np.float32)
    Y_expected = 1.0 / (1.0 + np.exp(-X))
    run_function(X, Y_expected, op.sigmoid, rtol=1e-4)


def test_tanh():
    X = np.random.randn(3, 5).astype(np.float32)
    Y_expected = 2 * (1.0 / (1.0 + np.exp(-2*X))) - 1.0
    run_function(X, Y_expected, op.tanh, rtol=1e-4)


def test_drelu_delta():
    X = np.random.randn(3, 5).astype(np.float32)
    A = 5*np.random.randn(3, 5).astype(np.float32)
    D = 5*np.random.randn(3, 5).astype(np.float32)
    D_expected = D * (A > 0)
    Dd = op.to_gpu(D)
    Yh = op.drelu_delta(D, A, X)
    assert_allclose(D_expected, D, rtol=1e-5, err_msg="CPU")
    Ad = op.to_gpu(A)
    Xd = op.to_gpu(X)
    op.drelu_delta(Dd, Ad, Xd)
    assert_allclose(D_expected, op.to_cpu(Dd), rtol=1e-5, err_msg="GPU")


def test_dtanh_delta():
    X = np.random.randn(3, 5).astype(np.float32)
    A = 5*np.random.randn(3, 5).astype(np.float32)
    D = 5*np.random.randn(3, 5).astype(np.float32)
    D_expected = D * (1 - A*A)
    Dd = op.to_gpu(D)
    Yh = op.dtanh_delta(D, A, X)
    assert_allclose(D_expected, D, rtol=1e-5, err_msg="CPU")
    Ad = op.to_gpu(A)
    Xd = op.to_gpu(X)
    op.dtanh_delta(Dd, Ad, Xd)
    assert_allclose(D_expected, op.to_cpu(Dd), rtol=1e-5, err_msg="GPU")


def test_dsigmoid_delta():
    X = np.random.randn(3, 5).astype(np.float32)
    A = 5*np.random.randn(30, 50).astype(np.float32)
    D = 5*np.random.randn(30, 50).astype(np.float32)
    D_expected = D * A*(1 - A)
    Dd = op.to_gpu(D)
    Yh = op.dsigmoid_delta(D, A, X)
    assert_allclose(D_expected, D, rtol=1e-5, err_msg="CPU")
    Ad = op.to_gpu(A)
    Xd = op.to_gpu(X)
    op.dsigmoid_delta(Dd, Ad, Xd)
    assert_allclose(D_expected, op.to_cpu(Dd), rtol=1e-5, err_msg="GPU")


def test_toplayer_delta():
    X = np.random.randn(3, 5).astype(np.float32)
    A = 5*np.random.randn(30, 50).astype(np.float32)
    D = 5*np.random.randn(30, 50).astype(np.float32)
    D_expected = D.copy()
    D_expected = A - D_expected
    Dd = op.to_gpu(D)
    Yh = op.toplayer_delta(A, D, X)
    assert_allclose(D_expected, Yh, rtol=1e-5, err_msg="CPU")
    Ad = op.to_gpu(A)
    Xd = op.to_gpu(X)
    Yhd = op.toplayer_delta(Ad, Dd, Xd)
    assert_allclose(D_expected, op.to_cpu(Yhd), rtol=1e-5, err_msg="GPU")

def test_softmax():
    X = np.random.randn(30, 50).astype(np.float32)
    E = np.exp(X)
    Y_expected = E / np.sum(E, axis=1).reshape(-1, 1)
    run_function(X, Y_expected, op.softmax, rtol=1e-4)

    X = 10000*np.random.randn(30, 50).astype(np.float32)
    Y = op.softmax(X)
    assert np.all(np.isfinite(Y))

    Y = op.softmax(op.to_gpu(X))
    assert np.all(np.isfinite(op.to_cpu(Y)))


def test_add_matvec():
    X = np.random.randn(3, 4).astype(np.float32)
    b1 = np.random.randn(4, 1).astype(np.float32)
    b2 = np.random.randn(3, 1).astype(np.float32)
    Y_expected1 = X + b1.T
    Y_expected2 = X + b2
    assert_allclose(Y_expected1, op.add_matvec(X, b1, 1))
    assert_allclose(Y_expected2, op.add_matvec(X, b2, 0))

    Xd = op.to_gpu(X)
    b1d = op.to_gpu(b1)
    b2d = op.to_gpu(b2)
    assert_allclose(Y_expected1, op.to_cpu(op.add_matvec(Xd, b1d, 1)))
    assert_allclose(Y_expected2, op.to_cpu(op.add_matvec(Xd, b2d, 0)))


def test_rand():
    X = np.empty((1000, 1000), dtype=np.float32)
    Y = op.rand_uniform_like(X)
    rtol = 1e-3
    assert (Y.mean() - 0.5) < rtol, "mean: %f" % Y.mean()
    assert Y.min() >= 0.0, "min: %f" % Y.min()
    assert Y.min() - 0.0 < rtol, "min: %f" % Y.min()
    assert Y.max() <= 1.0, "max: %f" % Y.max()
    assert Y.max() - 1.0 - rtol, "max: %f" % Y.max()

    Y = np.empty_like(X)
    out = op.rand_uniform_like(X, out=Y)
    assert out is Y
    assert (Y.mean() - 0.5) < rtol, "mean: %f" % Y.mean()
    assert Y.min() >= 0.0, "min: %f" % Y.min()
    assert Y.min() - 0.0 < rtol, "min: %f" % Y.min()
    assert Y.max() <= 1.0, "max: %f" % Y.max()
    assert Y.max() - 1.0 - rtol, "max: %f" % Y.max()

    Xd = op.to_gpu(X)
    Yd = gpuarray.empty_like(Xd)
    Y = op.to_cpu(op.rand_uniform_like(Xd))
    assert (Y.mean() - 0.5) < rtol, "mean: %f" % Y.mean()
    assert Y.min() >= 0.0, "min: %f" % Y.min()
    assert Y.min() - 0.0 < rtol, "min: %f" % Y.min()
    assert Y.max() <= 1.0, "max: %f" % Y.max()
    assert Y.max() - 1.0 - rtol, "max: %f" % Y.max()

    out = op.rand_uniform_like(Xd, out=Yd)
    assert out is Yd
    Y = op.to_cpu(Yd)
    assert (Y.mean() - 0.5) < rtol, "mean: %f" % Y.mean()
    assert Y.min() >= 0.0, "min: %f" % Y.min()
    assert Y.min() - 0.0 < rtol, "min: %f" % Y.min()
    assert Y.max() <= 1.0, "max: %f" % Y.max()
    assert Y.max() - 1.0 - rtol, "max: %f" % Y.max()


def test_rand_gaussian():
    X = np.empty((4000, 1000), dtype=np.float32)
    Y = op.rand_gaussian_like(X)
    rtol = 1e-3
    assert (Y.mean() - 0.0) < rtol, "mean: %f" % Y.mean()
    assert Y.std() - 1.0 < rtol, "std: %f" % Y.std()

    Y = op.rand_gaussian_like(X, mu=5.0, sigma=2.0)
    rtol = 1e-3
    assert (Y.mean() - 5.0) < rtol, "mean: %f" % Y.mean()
    assert Y.std() - 2.0 < rtol, "std: %f" % Y.std()

    Xd = op.to_gpu(X)
    Yd = gpuarray.empty_like(Xd)
    Y = op.to_cpu(op.rand_gaussian_like(Xd))
    rtol = 1e-2
    assert (Y.mean() - 0.0) < rtol, "mean: %f" % Y.mean()
    assert Y.std() - 1.0 < rtol, "std: %f" % Y.std()

    Y = op.to_cpu(op.rand_gaussian_like(Xd, mu=5.0, sigma=2.0))
    rtol = 1e-2
    assert (Y.mean() - 5.0) < rtol, "mean: %f" % Y.mean()
    assert Y.std() - 2.0 < rtol, "std: %f" % Y.std()


def test_max():
    X = np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 0.7]], dtype=np.float32)
    run_function_with_axis(X, X.max(0), X.max(1), X.max(), op.max)


def test_argmax():
    X = np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 0.7]], dtype=np.float32)
    run_function_with_axis(X, X.argmax(0), X.argmax(1), None, op.argmax)


def test_sum():
    #X = np.random.randn(1000, 400).astype(np.float32)
    X = np.array([[1.0, 2.0, 1.5], [3.0, 4.0, 0.75]], dtype=np.float32)
    run_function_with_axis(X, X.sum(0), X.sum(1), X.sum(), op.sum)


def test_mean():
    X = np.random.randn(100, 40).astype(np.float32)
    run_function_with_axis(X, X.mean(0), X.mean(1), X.mean(), op.mean, rtol=1e-4)


def test_std():
    X = 5*np.random.randn(100, 40).astype(np.float32)
    run_function_with_axis(X, X.std(0), X.std(1), X.std(), op.std, rtol=1e-4)


def test_crossentropy():
    X = np.random.rand(100, 10).astype(np.float32)
    O = np.random.rand(100, 10).astype(np.float32)
    X /= X.sum(1)[:, None]
    O /= O.sum(1)[:, None]
    Y_expected = -np.sum(X * np.log(O)) / X.shape[0]
    rtol=1e-4
    Y = np.empty_like(X)
    Yhr = op.cross_entropy(X, O)
    assert_allclose(Y_expected, Yhr, err_msg="CPU, no target", rtol=rtol)

    Xd = op.to_gpu(X)
    Od = op.to_gpu(O)
    Yd = op.cross_entropy(Xd, Od)
    assert_allclose(Y_expected, op.to_cpu(Yd), err_msg="GPU, no target", rtol=rtol)


def test_add_vec():
    x = 5.0 * np.random.randn(10).astype(np.float32)
    y = 10.0 * np.random.randn(10).astype(np.float32)
    x_orig = x.copy()
    alpha = 2.5
    z = x + alpha*y
    rtol = 1e-4

    op.add_vec(x, alpha, y)
    assert_allclose(z, x, err_msg="CPU", rtol=rtol)

    xd = op.to_gpu(x_orig)
    yd = op.to_gpu(y)
    op.add_vec(xd, alpha, yd)
    res = op.to_cpu(xd)
    assert_allclose(z, res, err_msg="GPU", rtol=rtol)

    x = x_orig.copy()
    alpha = 2.5
    beta = 0.5
    z = beta*x + alpha*y
    rtol = 1e-4

    op.add_vec(x, alpha, y, beta)
    assert_allclose(z, x, err_msg="CPU", rtol=rtol)

    xd = op.to_gpu(x_orig)
    yd = op.to_gpu(y)
    op.add_vec(xd, alpha, yd, beta)
    res = op.to_cpu(xd)
    assert_allclose(z, res, err_msg="GPU", rtol=rtol)


def test_l1reg():

    # NOTE: you could argue wether it's okay to "jump over zero"
    #       when applying both the regular gradient and the L1 gradient
    l1_penalty=0.005
    w = np.array( [3.0, 0.01, -0.01, 0.010, -0.010]).astype(np.float32)
    dw = np.array([2.9, 0.10, -0.10, 0.006, +0.006]).astype(np.float32)
    eta = 1.0

    nw = w + dw - l1_penalty*np.sign(w)
    expected = np.where(w > 0, np.maximum(0, nw), np.minimum(0, nw))
    y = np.empty_like(dw)
    op.add_vec_l1reg(w, dw, eta, l1_penalty, out=y)
    assert_allclose(expected, y)

    wd = op.to_gpu(w)
    dwd = op.to_gpu(dw)

    yd = op.to_gpu(np.empty_like(dw))
    op.add_vec_l1reg(wd, dwd, eta, l1_penalty, out=yd)
    assert_allclose(expected, op.to_cpu(yd))


def test_soft_threshold():
    l1_penalty=0.1
    X = np.array( [-3.0, 3.0, 0.2, -0.2, 0.01, -0.01]).astype(np.float32)
    E = np.array( [-2.9, 2.9, 0.1, -0.1, 0.00, -0.00]).astype(np.float32)
    run_function(X, E, op.soft_threshold, rtol=1e-4, alpha=l1_penalty)


def test_swaprows():
    n = 1270

    X = 5.0*np.random.randn(n, 1000).astype(np.float32)
    ytemp = np.array(range(X.shape[0]))[:, None]
    y = np.hstack((ytemp, ytemp, ytemp)).astype(np.float32)

    idx = list(range(X.shape[0]))
    idx = np.array(idx, dtype=np.int32)
    np.random.shuffle(idx)

    Xd = op.to_gpu(X)
    yd = op.to_gpu(y)
    Xoutd = gpuarray.empty_like(Xd)
    youtd = gpuarray.empty_like(yd)
    op.shuffle_rows(Xd, yd, (Xoutd, youtd), idx)

    X2 = op.to_cpu(Xoutd)
    y2 = op.to_cpu(youtd)

    assert_allclose(X[idx], X2)
    assert_allclose(y[idx], y2)


def test_reorderrows():
    n = 1270
    X = 5*np.random.randn(n, 1000).astype(np.float32)
    idx = list(range(X.shape[0]))
    np.random.shuffle(idx)
    Xd = op.to_gpu(X)
    Xoutd = gpuarray.empty_like(Xd)
    op.reorder_rows(Xd, idx, Xoutd)
    assert_allclose(X[idx], Xoutd.get())
    assert_allclose(X[idx], op.reorder_rows(X, idx))


def test_clip():
    X = 5*np.random.randn(30, 50).astype(np.float32)
    Y_expected = X.copy()
    Y_expected[X < 0] = 0.0
    Y_expected[X > 2] = 2.0
    run_function(X, Y_expected, op.clip, rtol=1e-4, minval=0.0, maxval=2.0)

    X = 5*np.random.randn(30, 50).astype(np.float32)
    Y_expected = X.copy()
    Y_expected[X < -1.0] = -1.0
    Y_expected[X > 0.25] = 0.25
    run_function(X, Y_expected, op.clip, rtol=1e-4, minval=-1.0, maxval=0.25)


def test_dot():
    X = 5*np.random.randn(30, 50).astype(np.float32)
    Y = 5*np.random.randn(50, 30).astype(np.float32)
    R = np.dot(X, Y)
    run_function(X, R, op.dot, rtol=1e-2, with_inplace_test=False, b=Y)

    X = 5*np.random.randn(30, 50).astype(np.float32)
    Y = 5*np.random.randn(50, 30).astype(np.float32)
    R = 2*np.dot(X, Y)
    run_function(X, R, op.dot, rtol=1e-2, with_inplace_test=False, b=Y, alpha=2.0)


def test_dense_gemm():
    A = np.random.randn(30, 40).astype(np.float32)
    B = np.random.randn(40, 50).astype(np.float32)
    X = np.ones((30, 50), np.float32)
    X_exp = np.dot(A, B) + X
    op.add_dot(A, B, X, beta=1.0)
    assert_allclose(X, X_exp)
    Ad = op.to_gpu(A)
    Bd = op.to_gpu(B)
    Xd = op.to_gpu(X)
    op.add_dot(A, B, X, beta=1.0)
    assert_allclose(op.to_cpu(Xd), X_exp)

    A = np.random.randn(40, 30).astype(np.float32)
    B = np.random.randn(40, 50).astype(np.float32)
    X = np.ones((30, 50), np.float32)
    X_exp = np.dot(A.T, B) + X
    op.add_dot(A, B, X, transA=True, beta=1.0)
    assert_allclose(X, X_exp)
    Ad = op.to_gpu(A)
    Bd = op.to_gpu(B)
    Xd = op.to_gpu(X)
    op.add_dot(A, B, X, transA=True, beta=1.0)
    assert_allclose(op.to_cpu(Xd), X_exp)

    A = np.random.randn(30, 40).astype(np.float32)
    B = np.random.randn(50, 40).astype(np.float32)
    X = np.ones((30, 50), np.float32)
    X_exp = np.dot(A, B.T) + X
    op.add_dot(A, B, X, transB=True, beta=1.0)
    assert_allclose(X, X_exp)
    Ad = op.to_gpu(A)
    Bd = op.to_gpu(B)
    Xd = op.to_gpu(X)
    op.add_dot(A, B, X, transB=True, beta=1.0)
    assert_allclose(op.to_cpu(Xd), X_exp)

    A = np.random.randn(40, 30).astype(np.float32)
    B = np.random.randn(50, 40).astype(np.float32)
    X = np.ones((30, 50), np.float32)
    X_exp = np.dot(A.T, B.T) + X
    op.add_dot(A, B, X, transA=True, transB=True, beta=1.0)
    assert_allclose(X, X_exp)
    Ad = op.to_gpu(A)
    Bd = op.to_gpu(B)
    Xd = op.to_gpu(X)
    op.add_dot(A, B, X, transA=True, transB=True, beta=1.0)
    assert_allclose(op.to_cpu(Xd), X_exp)


def test_randomly_replace_elements():
    for val in (0.0, 0.5, 5):
        for p in (0.1, 0.2, 0.5, 0.75, 0.99):
            X = np.random.normal(size=(1024, 2048)).astype(np.float32)
            Xd = op.to_gpu(X)
            Xr, M = op.randomly_replace_elements(X, p, val)
            assert(Xr is X)
            assert_almost_equal((X == val).mean(), p, decimal=2,
                                err_msg="val: %.1f p: %.1f" % (val, p))
            assert_almost_equal(M.mean(), 1-p, decimal=2,
                                err_msg="M val: %.1f p: %.1f" % (val, p))

            Xrd, Md = op.randomly_replace_elements(Xd, p, val)
            assert(Xrd is Xd)
            assert_almost_equal(op.to_cpu(op.mean(Xd == val)), p, decimal=2,
                                err_msg="val: %.1f p: %.1f (gpu)" % (val, p))
            assert_almost_equal(op.to_cpu(op.mean(Md)), 1-p, decimal=2,
                                err_msg="M val: %.1f p: %.1f (gpu)" % (val, p))


def test_cublas_bug():
    '''
    The SGEMM call would cause all calls after it to fail for some unknown
    reason. Likely this is caused swaprows causing memory corruption.

    NOTE: this was confirmed by nvidia to be a bug within CUDA, and should be
          fixed in CUDA 6.5
    '''
    from pycuda.driver import Stream
    from skcuda.cublas import cublasSgemm
    from skcuda.misc import _global_cublas_handle as handle

    n = 131

    s = slice(128, n)
    X = gpuarray.to_gpu(np.random.randn(n, 2483).astype(np.float32))
    a = gpuarray.empty((X.shape[1], 3), dtype=np.float32)
    c = gpuarray.empty((a.shape[0], X.shape[1]), dtype=np.float32)
    b = gpuarray.empty_like(X)

    m, n = a.shape[0], b[s].shape[1]
    k = a.shape[1]
    lda = m
    ldb = k
    ldc = m
    #cublasSgemm(handle, 0, 0, m, n, k, 0.0, b.gpudata, lda, a.gpudata, ldb, 0.0, c.gpudata, ldc)
    cublasSgemm(handle, 'n', 'n', m, n, k, 1.0, b[s].gpudata, lda, a.gpudata, ldb, 0.0, c.gpudata, ldc)
    #print handle, 'n', 'n', m, n, k, 1.0, b[s].gpudata, lda, a.gpudata, ldb, 0.0, c.gpudata, ldc

    #gpuarray.dot(d, Xoutd[s])
    #op.sgemm(a, b[s], c)

    stream = Stream()
    stream.synchronize()


def test_nan_in_toplayer_delta():
    size = (200, 10)
    X = np.random.normal(size=size).astype(np.float32, order="c")
    A = op.sigmoid(X)
    Y = np.random.binomial(1.0, p=0.5, size=size).astype(np.float32)
    M = np.random.binomial(1.0, p=0.9, size=size).astype(np.float32)
    Y[~M.astype(np.bool)] = np.nan
    Y_orig = Y.copy()
    D = M * (A - Y)
    D[~M.astype(np.bool)] = 0.0

    Y = op.toplayer_delta(A, Y, X)
    assert_allclose(Y, D)

    Yd = op.to_gpu(Y_orig)
    Ad = op.to_gpu(A)
    Xd = op.to_gpu(X)
    Yd = op.toplayer_delta(Ad, Yd, Xd)
    assert_allclose(Yd.get(), D)


def test_swapaxes01():
    X = np.arange(5*7*3).reshape(5, 7, 3).astype(np.float32)
    Y_expected = np.swapaxes(X, 0, 1).copy()
    run_function(X, Y_expected, op.swapaxes01, with_inplace_test=False)

    def run_swapaxes_twice(X, out=None):
        Y = op.swapaxes01(X)
        X2 = op.swapaxes01(Y, out=out)
        return X2

    run_function(X, X, run_swapaxes_twice, with_inplace_test=False)


def test_to_onehot():
    n_samples = 13
    n_classes = 7

    labels = np.random.choice(n_classes, n_samples).astype(np.uint16)
    expected = np.zeros((n_samples, n_classes), dtype=np.float32)
    for i, c in enumerate(labels):
        expected[i, c] = 1
    run_function(labels, expected, op.to_onehot, with_inplace_test=False, n_classes=n_classes)


def test_sequence_to_tensor():
    x =  np.asarray([0, 1, 3, 1, 4, 4, 2, 3], dtype=np.uint16)
    seqlen = 3
    n_samples = x.shape[0]-seqlen
    n_classes = x.max()+1
    y_exp = np.zeros((seqlen, n_samples, n_classes), np.float32)
    for t in range(seqlen):
        for i in range(n_samples):
            cls = x[t+i]
            y_exp[t, i, cls] = 1
    run_function(x, y_exp, op.sequence_to_tensor, with_inplace_test=False,
    seqlen=seqlen,  n_classes=x.max()+1)
