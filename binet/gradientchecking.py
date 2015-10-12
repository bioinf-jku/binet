# -*- coding: utf-8 -*-
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import
'''
Code for gradient checking

Copyright © 2013-2015 Thomas Unterthiner
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''
import numpy as np
from scipy import optimize
from binet import op


def serialize(weights, bias):
    return np.concatenate(
        [np.concatenate([weights[i].flatten(), bias[i].flatten()])
        for i in range(len(weights))])


def unserialize(params, net):
    i = 0
    for l in net.layers:
        n = l.W.size
        l.W[:] = params[i:i+n].reshape(l.W.shape)
        i += n
        n = l.b.size
        l.b[:] = params[i:i+n].reshape(l.b.shape)
        i += n
    return net


def cross_entropy_loss(w, x, y, net):
    unserialize(w, net)
    a = net.forward_pass(x)
    retval = op.cross_entropy(y, a)
    return retval


def squared_error(w, x, y, net):
    unserialize(w, net)
    a = net.forward_pass(x)
    return 0.5*((y-a)**2).sum() / y.shape[0]


def grad(w, x, y, net):
    unserialize(w, net)
    a = net.forward_pass(x)
    net.backward_pass(a, y)
    dw = [l.dW for l in net.layers]
    db = [l.db for l in net.layers]
    g = serialize(dw, db)
    return g


def check_gradient(loss_func, net, n_samples, n_features, n_outputs):
    x = 3.0*np.random.normal(size=(n_samples, n_features)).astype(net.dtype)
    y = np.random.multinomial(1, [1.0 / n_outputs]*n_outputs, size=n_samples).astype(net.dtype)
    w0 = serialize(net.weights, net.bias)
    return optimize.check_grad(loss_func, grad, w0, x, y, net)


if __name__ == "__main__":
    from binet.neuralnet import NeuralNet

    n_samples = 50
    n_features = 2
    n_outputs = 3

    # sigmoid classification w softmax
    net = NeuralNet((n_features, 4, n_outputs), dtype=np.double,
                    activation="sigmoid", output="softmax", loss="crossentropy")
    d = check_gradient(cross_entropy_loss, net, n_samples, n_features, n_outputs)
    print("sigmoid->softmax + CE: %f" % d)

    net = NeuralNet((n_features, 4, n_outputs), dtype=np.double,
                    activation="relu", output="softmax", loss="crossentropy")
    d = check_gradient(cross_entropy_loss, net, n_samples, n_features, n_outputs)
    print("relu->softmax + CE: %f" % d)

    net = NeuralNet((n_features, 4, n_outputs), dtype=np.double,
                    activation="tanh", output="softmax", loss="crossentropy")
    d = check_gradient(cross_entropy_loss, net, n_samples, n_features, n_outputs)
    print("tanh->softmax + CE: %f" % d)

    net = NeuralNet((n_features, 4, n_outputs), dtype=np.double,
                    activation="tanh", output="linear", loss="squarederror")
    d = check_gradient(squared_error, net, n_samples, n_features, n_outputs)
    print("tanh->linear + se: %f" % d)

    net = NeuralNet((n_features, 4, n_outputs), dtype=np.double,
                    activation="relu", output="linear", loss="squarederror")
    d = check_gradient(squared_error, net, n_samples, n_features, n_outputs)
    print("relu->linear + se: %f" % d)

    net = NeuralNet((n_features, 10, 9, n_outputs), dtype=np.double,
                    activation="relu", output="softmax", loss="crossentropy")
    d = check_gradient(cross_entropy_loss, net, n_samples, n_features, n_outputs)
    print("relu->relu->softmax + CE: %f" % d)

    net = NeuralNet((n_features, 15, 16, 14, n_outputs), dtype=np.double,
                    activation="relu", output="softmax", loss="crossentropy")
    d = check_gradient(cross_entropy_loss, net, n_samples, n_features, n_outputs)
    print("reul->relu->relu->softmax + CE: %f" % d)
