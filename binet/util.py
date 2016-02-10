# -*- coding: utf-8 -*-
'''
Often-needed functions when using binet

Copyright © 2013-2015 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)
'''

from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division
from __future__ import absolute_import

import sys
if sys.version_info < (3,):
    range = xrange
    import cPickle as pickle
else:
    import pickle

import numpy as np

import time
import os
import gc
import logging
import warnings
import copy

# Importing matplotlib might fail under special conditions
# e.g. when using ssh w/o X11 forwarding
try:
    import matplotlib.pyplot as plt
except ImportError:
    warnings.warn("matplotlib unavailable")

def generate_slices(n, size, ignore_last_minibatch_if_smaller=False):
    """Generates slices of given size up to n"""
    start, end = 0, 0
    for pack_num in range(int(n / size)):
        end = start + size
        yield slice(start, end, None)
        start = end
    # last slice might not be a full batch
    if not ignore_last_minibatch_if_smaller:
        if end < n:
            yield slice(end, n, None)


def plot_images(data, nrows, ncols, is_color=False, axis=None,
                local_norm="maxabs", **kwargs):
    ''' Plots several images stored in the rows of data.'''
    nchannels = 3 if is_color else 1
    ppi = int(np.sqrt(data.shape[-1]/nchannels) + 2) # pixel per image +2 for borders
    imgshape = (nrows*ppi, ncols*ppi, nchannels)
    # make sure border is black
    img = {"maxabs": lambda s: (data.min() / np.abs(data).max()) * np.ones(imgshape, dtype=data.dtype),
           "minmax": lambda s: np.zeros(imgshape, dtype=data.dtype),
           "none":   lambda s: np.ones(imgshape, dtype=data.dtype)*data.min()
            }[local_norm.lower()](None)
    if len(data.shape) < 3:
        data = data.reshape(data.shape[0], nchannels, ppi-2, ppi-2)
    n = min(nrows*ncols, data.shape[0])
    normfunc = {"maxabs": lambda d: d / np.abs(d).max(),
                "minmax": lambda d: (d - d.min()) / d.ptp(), # normalize to [0, 1]
                "none":   lambda d: d
    }[local_norm.lower()]
    idx = 0
    for r in range(nrows):
        for c in range(ncols):
            if idx >= n:
                break
            d = np.rollaxis(data[idx, ], 0, 3)
            d = normfunc(d)
            img[r*ppi+1:(r+1)*ppi-1, c*ppi+1:(c+1)*ppi-1] = d
            idx += 1
    if axis==None:
        fig = plt.figure(facecolor="black", **kwargs)
        fig.subplots_adjust(hspace=0, top=1, bottom=0, wspace=0, left=0, right=1)
        axis = fig.gca()
    else:
        fig = None
    if is_color:
        axis.imshow(img, interpolation="none")
    else:
        axis.imshow(img.reshape(nrows*ppi, ncols*ppi), interpolation="none", cmap="Greys_r")
    axis.axis("off")
    return fig


def heuristic_svm_c(x):
    ''' Heuristic for setting the C for linear SVMS proposed by Thorsten Joachims.'''
    c = 0
    n = x.shape[0]
    for i in range(n):
        c += np.sqrt(x[i, ].dot(x[i, ]))
    c /= n
    return 1.0 / c


def plot_learning_curves(net, start_idx=5, end_idx=None,
                         min_error=np.log(np.finfo(np.float32).tiny),
                        *args, **kwargs):
    if end_idx is None or end_idx > net.statistics.shape[0]:
        end_idx = net.statistics.shape[0]

    if end_idx - start_idx <= 0:
        warnings.warn("Not enough data to plot learning curves")
        return

    data = net.statistics.ix[start_idx:end_idx]
    fig = plt.figure(*args, **kwargs)
    ax1 = plt.gca()
    np.log10(data[["train_error", "val_error"]]).plot(ax=ax1, legend=False)
    ax1.set_xlabel("epoch")
    ax1.set_ylabel("Cross-Entropy Error (log10)")

    ax2 = ax1.twinx()
    colcyc = ax2._get_lines.color_cycle # we need to jump 2 colors
    col = [next(colcyc), next(colcyc), next(colcyc)]
    data[['val_score']].plot(ax=ax2, color=col[2], linestyle=":", legend=False)
    ax2.set_ylabel("Validationset Accuracy", color=col[2])

    # we need to draw the legend separately, otherwise each axis would create
    # its own legend
    handles, labels = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    handles += h2
    labels += l2
    fig.legend(handles, labels, loc="lower left", prop={'size':9})
    fig.tight_layout
    return fig


def train(net, dataset, fname=None, skip_output=25,
          show_plots=False, use_gpu=True, **kwargs):
    ''' Trains a neural network on the given dataset.

    If desired, the log-statements during training can be buffered into a
    StringIO object. This has the drawback that the output is only visible
    once the net has been fully trained, but it allows to only print only every
    n-th message.

    Parameters
    ----------
    net: the neural net.
    dataset: tuple containing 'trainx', 'trainy', 'validx', 'validy'
    fname: file-name in which to store the (pickled) network after training.
           The file will be stored in the 'data' subfolder of the CWD.
    skip_output: how many lines of output to skip between two lines that
                 will actually be printed.
    show_plots: If True, plot the first 256 weights of the lowest layer.
    use_gpu: if True, use gnumpy to run the code on the GPU.
    **kwargs: additional parameters for the `plotImages` cool when
              `plot_weights=True`.
    '''
    from binet import op
    if use_gpu:
        gc.collect()
        if not op._IS_CUDA_INITIALIZED:
            logger = logging.getLogger(__name__)
            logger.warn("CUDA not initialized, initializing GPU 0")
            op.init_gpu(0)

        X, y, Xvalid, yvalid = [op.to_gpu(d) for d in dataset]
        net = op.to_gpu(net)
    else:
        X, y, Xvalid, yvalid = dataset
    try:
        init_out = net.transform(X)
        init_err = net._get_loss(y, init_out)
        net.track_progress(time.time(), init_err, X, y, Xvalid, yvalid)
        net.fit(X, y, Xvalid, yvalid, skip_output=skip_output)
        #if net.verbose and net.current_epoch % skip_output != 0: # make sure we show the last line
        #    net.track_progress(time.time(), -1, X, y, Xvalid, yvalid)
    except KeyboardInterrupt:
        print("Intercepted KeyboardInterrupt, stopping... current status:")
        net.track_progress(time.time(), -1, X, y, Xvalid, yvalid)
        net.statistics = net.statistics[:-1] # we just added an invalid point
    finally:
        net = op.to_cpu(net)
        if fname:
            if not os.path.exists("data"):
                warnings.warn("creating 'data' directory to store pickled net")
                os.mkdir("data")
            with open(os.path.join("data", fname), "wb") as f:
                pickle.dump(net, f, -1)
        if show_plots:
            plot_images(net.weights[0], 16, 16, **kwargs)
            plot_learning_curves(net, **kwargs)
    return net


def train_ensemble(prototype_net, dataset, outfile=None, n_nets=10, use_gpu=True):
    ''' Trains a given number of networks on a given dataset.

    All networks will be clones of the given prototoype, and they will all
    be pickled into the given outfile.'''
    from binet import op
    if use_gpu:
        gc.collect()
        if not op._IS_CUDA_INITIALIZED:
            logger = logging.getLogger(__name__)
            logger.warn("CUDA not initialized, initializing GPU 0")
            op.init_gpu(0)

        X, y, Xvalid, yvalid = [op.to_gpu(d) for d in dataset]
        prototype_net = op.to_gpu(prototype_net)
    else:
        X, y, Xvalid, yvalid = dataset
    if outfile is not None:
        f = open(outfile, "wb")
    nets = []
    try:
        for i in range(n_nets):
            prototype_net.reset()
            if use_gpu:
                prototype_net = op.to_gpu(prototype_net)
            prototype_net.fit(X, y, Xvalid, yvalid)
            prototype_net = op.to_cpu(prototype_net)
            nets.append(copy.deepcopy(prototype_net))
            if outfile is not None:
                pickle.dump(prototype_net, f, -1)
    finally:
        if outfile is not None:
            f.close()
    return nets


def load_ensemble(fn):
    nets = []
    with open(fn) as f:
        try:
            while f:
                nets.append(pickle.load(f))
        except EOFError:
            return nets


def print_system_information(additional_modules=[]):
    '''Prints general system information.

    Prints host information as well as version information about some of the
    more important packages. This is useful in IPython notebooks.'''

    import sys, os, datetime, platform
    host_info = (platform.node(), platform.platform())
    print("Host:               ", "%s: %s" % host_info)
    print("Date:               ", str(datetime.datetime.now()))
    print("Python version:     ", sys.version.replace("\n", "\n" +  " "*21))

    repo_version = str(os.popen("git log | head -1").readline().strip())
    if not repo_version.startswith("fatal:"):
        print("repository version: ", repo_version)

    print("\nloaded modules:")

    # make sure most important modules are here, even if we only imported
    # some submodules
    import binet, numpy, scipy
    modlist = ['scipy', 'numpy', 'sklearn', 'IPython', 'matplotlib',
               'binet', 'pandas', 'joblib']
    modlist.extend(additional_modules)
    mod = [sys.modules[m]for m in  modlist if m in sys.modules]
    mod.sort(key = lambda x: x.__name__)
    for m in mod:
        try:
             print("\t", m.__name__, m.__version__)
        except AttributeError:
            pass
