#!/usr/bin/python
# -*- coding: utf-8 -*-
'''
Dataset handling within binet.

Copyright © 2013-2015 Thomas Unterthiner.
Licensed under GPL, version 2 or a later (see LICENSE.rst)

binet stores datasets as HDF5 files. A dataset is comprised of 6 matrices:
trainx, trainy, validx, validy, testx, testy

These are usually stored as float-values, with the labels (y-values) in a
one-hot encoding.

NOTE: This file can be executed. It then converts datasets from their original
format into HDF5 files.

    Usage:  datasets.py (mnist | norb | cifar10) [directory]
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

import struct
import gzip
import zipfile
import os
import scipy
import platform
import logging
import gc

import numpy as np
import pandas as pd

from scipy import io

try:
    import h5py
except ImportError:
    import warnings
    warnings.warn("h5py unavailable")

# some machine specific paths for bioinf@jku machines
__datadir = {'tomlap': '/media/scratch/data',
             'blucomp': '/media/scratch/data'}
_DATA_DIRECTORY = __datadir.get(platform.node(), os.path.expanduser("~/data"))


def load_dataset(dataset_name, return_testset=False, dtype=np.float32, revert_scaling=False):
    '''Loads a dataset, given the filename of the HDF5 file.

    Returns 4 tuple of X, y, Xvalid, yvalid)
    '''
    if not dataset_name.endswith(".hdf5"):
        fname = os.path.join(_DATA_DIRECTORY, dataset_name + ".hdf5")
    else:
        fname = os.path.join(_DATA_DIRECTORY, dataset_name)

    # try to create standard datset if it doesn't exist yet
    if not os.path.exists(fname):
        createfuncs = {
            'mnist': _create_mnist,
            'norb': _create_norb,
            'cifar10': _create_cifar10_flat,
            'cifar10_img': _create_cifar10_img,
            'mnist_basic': _create_mnist_basic,
            'mnist_bgimg': _create_mnist_bgimg,
            'mnist_bgrand': _create_mnist_bgrand,
            'mnist_rot': _create_mnist_rot,
            'rectangles': _create_rectangles,
            'convex': _create_convex,
            'covertype': _create_covertype,
            'enwik8': _create_enwik8,
            'tox21': _create_tox21}
        cf = createfuncs.get(dataset_name, None)
        if cf is not None:
            l = logging.getLogger(__name__)
            l.warning("%s does not exist, trying to create it" % fname)
            cf(_DATA_DIRECTORY)

    if not os.path.exists(fname):
        raise RuntimeError("File %s does not exist" % fname)
    with h5py.File(fname) as dataset:
        if dataset_name == "enwik8":
            ds_keys = ['train', 'valid', 'test']
        else:
            ds_keys = ['trainx', 'trainy', 'validx', 'validy']
            if return_testset:
                ds_keys.extend(['testx', 'testy'])

        data = []
        s = dataset['scale'][:] if 'scale' in dataset else 1.0
        c = dataset['center'][:] if 'center' in dataset else 0.0
        for  k in ds_keys:
            if k.endswith('x') and revert_scaling:
                data.append(((dataset[k][:] * s)+c).astype(dtype))
            else:
                data.append(dataset[k][:].astype(dtype))
    gc.collect()
    return data


def _download_file(urlbase, fname, destination_dir):
    if not os.path.exists(destination_dir):
        os.mkdir(destination_dir)
    if sys.version_info < (3,):
        from urllib import urlretrieve
    else:
        from urllib.request import urlretrieve
    url = urlbase + fname
    dst = os.path.join(destination_dir, fname)
    if not os.path.exists(dst):
        logging.getLogger(__name__).info("downloading %s to %s" % (url, dst))
        urlretrieve(url, dst)
    return dst


def _to_one_hot_encoding(labels, dtype=np.float64):
    labels = labels.reshape((labels.shape[0], 1))
    '''Creates a one-hot encoding of the labels.'''
    from sklearn.preprocessing import OneHotEncoder
    enc = OneHotEncoder(dtype=dtype)
    return enc.fit_transform(labels).toarray()


def _shuffle(data, labels):
    ''' Shuffles the data and the labels.'''
    np.random.seed(42)  # Make sure the same file is produced on each machine
    idx = np.array(range(data.shape[0]))
    np.random.shuffle(idx)
    data = data[idx, :]
    labels = labels[idx, :]
    return data, labels, idx


def _read_mnist_image(filename):
    with gzip.open(filename) as f:
        buf = f.read(16)
        magic, n_items, xsize, ysize = struct.unpack(">iiii", buf)
        assert(magic == 2051) # magic number
        n_features = xsize*ysize
        data = np.zeros((n_items, n_features), dtype=np.uint8)
        for i in range(n_items):
            buf = f.read(n_features)
            x = struct.unpack("B"*n_features, buf)
            data[i, :] = x
    return data


def _read_mnist_label(filename):
    with gzip.open(filename) as f:
        buf = f.read(8)
        magic, n_items = struct.unpack(">ii", buf)
        assert(magic == 2049) # magic number
        data = np.zeros(n_items, dtype=np.uint8)
        buf = f.read(n_items)
        data[:] = struct.unpack("B"*n_items, buf)
    return data.reshape(-1, 1)


def _read_norb_data(filename):
    with gzip.open(filename) as f:
        buf = f.read(8)
        magic, ndims = struct.unpack("<ii", buf)
        if magic == 0x1e3d4c55:
            dt = np.dtype(np.uint8)
        elif magic == 0x1e3d4c54:
            dt = np.dtype(np.uint32)
        else:
            assert(False)
        n = max(ndims, 3)
        buf = f.read(n * 4)
        dims = struct.unpack('<' + ('i'*n) , buf)
        nitems = dims[0]
        nfeatures = int(np.prod(dims[1:]))
        data = np.empty((nitems, nfeatures), dtype=dt.type)

        # we have to iterate here, as doing it all at once might cause a MemoryError
        for i in range(nitems):
            buf = f.read(nfeatures*dt.itemsize)
            data[i] = struct.unpack(dt.char*nfeatures, buf)
    return data


def _store(data, filename, other=None):
    #
    # Note: deactivating compression got a MASSIVE boost in read-speed.
    # Our only compression-choice was gzip, as rhdf5 (R implementation)
    # could not handle LZO.
    # without compression, CIFAR10 can be read in <1 second in R (filesize ~750MB)
    # with GZIP, no matter what compression level, the times were ~40s.
    # (even though GZIP with compression_opts = 0 resulted in a file of 750MB)
    # (compression_opts = 9 reached ~250 MB)
    #
    logging.getLogger(__name__).info("saving into %s ..." % filename)
    with h5py.File(filename, "w") as f:
        for i in range(len(data)):
            f.create_dataset(data[i][0] + "x", data=data[i][1])
            f.create_dataset(data[i][0] + "y", data=data[i][2])#, compression="gzip", compression_opts = 0)
        if other:
            for k in other:
                f.create_dataset(k, data=other[k])


def _process_and_store(data, filename, other=None, rescale=False, dtype=np.float32):
    '''Shuffles, converts and stores the data.

    Shuffles training and testset, converts the data to np.float64 and stores it.
    `other` can be dictionary of additional data to store.

    data is expected to be a list of datasets, where each dataset is a list of
    [name, data, labels]. I.e. a normal train/testset split would be
    data = [ ['train', traindata, trainlabels], ['test', testdata, testlabels]]
    '''
    logger = logging.getLogger(__name__)
    logger.info("shuffling...")
    for i in range(len(data)):
        data[i][1], data[i][2], _ = _shuffle(data[i][1], data[i][2])
    logger.info("converting...")
    for i in range(len(data)):
        data[i][1] = data[i][1].astype(dtype)
        data[i][2] = _to_one_hot_encoding(data[i][2], dtype=dtype)
    if rescale:
        s = data[0][1].max()  # scale based on training set
        for i in range(len(data)):
            data[i][1] /= s
        if other is None:
            other = {}
        other['scale'] = s*np.ones(data[0][1].shape[1])
    _store(data, filename, other)
    gc.collect()


def _split_dataset(data, labels, fraction):
    """ Splits a dataset into two set, with the first part
        obtaining fraction % of the data."""
    n = int(data.shape[0] * fraction + 0.5)
    idx = np.random.choice(range(data.shape[0]), n, replace=False)
    return (data[idx, ], labels[idx],
            np.delete(data, idx, 0), np.delete(labels, idx, 0))


def _create_mnist(directory):
    ''' MNIST dataset from yann.lecun.com/exdb/mnist/  '''
    from os.path import join

    os.chdir(directory)
    logger = logging.getLogger(__name__)
    logger.info("reading data...")
    urlbase = 'http://yann.lecun.com/exdb/mnist/'
    files = ['train-images-idx3-ubyte.gz', 'train-labels-idx1-ubyte.gz',
             't10k-images-idx3-ubyte.gz', 't10k-labels-idx1-ubyte.gz']
    destdir = join(directory, "raw")
    for fname in files:
         _download_file(urlbase, fname, destdir)
    trainx = _read_mnist_image(join(destdir, "train-images-idx3-ubyte.gz"))
    trainy = _read_mnist_label(join(destdir, "train-labels-idx1-ubyte.gz"))
    testx = _read_mnist_image(join(destdir, "t10k-images-idx3-ubyte.gz"))
    testy = _read_mnist_label(join(destdir, "t10k-labels-idx1-ubyte.gz"))

    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, join(directory, "mnist.hdf5"), rescale=True)


def _create_norb(directory):
    '''Small NORB dataset from www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/ '''

    urlbase = "http://www.cs.nyu.edu/~ylclab/data/norb-v1.0-small/"
    dst = os.path.join(directory, "raw")
    trainx = _read_norb_data(_download_file(urlbase,
        'smallnorb-5x46789x9x18x6x2x96x96-training-dat.mat.gz', dst))
    trainy = _read_norb_data(_download_file(urlbase,
        'smallnorb-5x46789x9x18x6x2x96x96-training-cat.mat.gz', dst))
    traini = _read_norb_data(_download_file(urlbase,
        'smallnorb-5x46789x9x18x6x2x96x96-training-info.mat.gz', dst))
    testx = _read_norb_data(_download_file(urlbase,
        'smallnorb-5x01235x9x18x6x2x96x96-testing-dat.mat.gz', dst))
    testy = _read_norb_data(_download_file(urlbase,
        'smallnorb-5x01235x9x18x6x2x96x96-testing-cat.mat.gz', dst))

    # instead of assigning the validation set randomly, we pick one of the
    # "instances" of the training set. This is much better than doing it randomly!
    fold = traini[:, 0].ravel()
    vi = (fold == 4)  # let's make instance 4 the validation-instance
    #print vi.sum()
    validx, trainx = trainx[vi], trainx[~vi]
    validy, trainy = trainy[vi], trainy[~vi]
    #print validx.shape, trainx.shape
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "norb.hdf5"), rescale=True)


def _create_norb_downsampled(directory):
    if not os.path.exists(os.path.join(directory, "norb.hdf5")):
        _create_norb(directory)
    def downsample(X):
        Xd = np.empty((X.shape[0], 2048))
        for i, x in enumerate(X):
            y = scipy.misc.imresize(x.reshape(96*2, 96), 1.0 / 3.0, "bicubic")
            Xd[i] = y.ravel()
        return Xd
    tmp = h5py.File(os.path.join(directory, "norb.hdf5"))
    trainx = downsample(tmp['trainx'])
    validx = downsample(tmp['validx'])
    testx = downsample(tmp['testx'])
    data = [['train', trainx,  tmp['trainy']],
            ['valid', validx,  tmp['validy']],
            ['test', testx,  tmp['testy']]]
    _store(data, os.path.join(directory, "norb_downsampled.hdf5"))


def _load_cifar10(directory):
    logger = logging.getLogger(__name__)
    logger.info("reading CIFAR10 data...")
    fname = _download_file("http://www.cs.toronto.edu/~kriz/",
                          "cifar-10-binary.tar.gz",
                          os.path.join(directory, "raw"))
    import tarfile
    with tarfile.open(fname) as tf:
        filemembers = tf.getmembers()
        trainx = np.zeros((0, 3072))
        trainy = np.zeros((0,), dtype=np.uint8)
        files = [f.name for f in filemembers if "data_batch" in f.name]
        files.sort()

        def _read_file(fn):
            f = tf.extractfile(fn)
            tmp = np.frombuffer(f.read(), np.uint8).reshape(-1, 3073)
            return tmp[:, 0].reshape(-1, 1), tmp[:, 1:].reshape(-1, 3*32*32)

        # save last batch as validation
        traindata = [_read_file(fn) for fn in files[0:len(files)-1]]
        y_tr = np.vstack([t[0] for t in traindata])
        x_tr = np.vstack([t[1] for t in traindata])

        y_va, x_va = _read_file(files[-1])
        y_te, x_te =  _read_file('cifar-10-batches-bin/test_batch.bin')
        return x_tr, y_tr, x_va, y_va, x_te, y_te


def _create_cifar10_flat(directory):
    ''' CIFAR-10, from www.cs.toronto.edu/~kriz/cifar.html.'''
    x_tr, y_tr, x_va, y_va, x_te, y_te = _load_cifar10(directory)

    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    dst = os.path.join(directory, "cifar10.hdf5")
    _process_and_store(data, dst, rescale=True)
    #imshow(np.rot90(traindata[882, ].reshape((3, 32, 32)).T), origin="lower")


def _create_cifar10_img(directory):
    ''' CIFAR-10 in nbatches x width x height x channels format
    from www.cs.toronto.edu/~kriz/cifar.html.'''
    x_tr, y_tr, x_va, y_va, x_te, y_te = _load_cifar10(directory)
    x_tr, x_va, x_te = [x.reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
                        for x in (x_tr, x_va, x_te)]

    data = [['train', x_tr, y_tr],
            ['valid', x_va, y_va],
            ['test', x_te, y_te]]
    dst = os.path.join(directory, "cifar10_img.hdf5")
    _store(data, dst)
    #imshow(np.rot90(traindata[882, ].reshape((3, 32, 32)).T), origin="lower")


def _handle_larochelle_icml2007(directory, fn, train_data_file, test_data_file,
                                rotate_images=True):
    '''Basic procedure to load the datasets from Larochelle et al., ICML 2007.
    Unfortunately the structure of the datasets differs sometimes,
    so we need this abstraction.

    fn = name of the zip file (w/o extension)
    train_data_file: name of the training set file within the archive
    test_data_file: name of the test set file within the archive
    rotate_images: rotate images (needed if file is in column-major format)
    '''
    import zipfile
    urlbase = "http://www.iro.umontreal.ca/~lisa/icml2007data/"
    dst = os.path.join(directory, "raw")
    f = _download_file(urlbase, '%s.zip' % fn, dst)
    with zipfile.ZipFile(f) as zf:
        tmp = np.loadtxt(zf.open(train_data_file))
        trainx, trainy = tmp[:, :-1].copy(), tmp[:, -1].copy()
        tmp = np.loadtxt(zf.open(test_data_file))
        testx, testy = tmp[:, :-1].copy(), tmp[:, -1].copy()
        trainy = trainy.reshape((-1, 1))
        testy = testy.reshape((-1, 1))
        if rotate_images:
            n = int(np.sqrt(trainx.shape[1]))
            trainx = np.rollaxis(trainx.reshape(trainx.shape[0], n, n), 2, 1)
            trainx = trainx.reshape(-1, n*n)
            testx = np.rollaxis(testx.reshape(testx.shape[0], n, n), 2, 1)
            testx = testx.reshape(-1, n*n)
        return trainx, trainy, testx, testy


def _create_mnist_basic(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory, "mnist",
        'mnist_train.amat', 'mnist_test.amat', rotate_images=False)
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "mnist_basic.hdf5"), rescale=True)


def _create_mnist_bgimg(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory,
        "mnist_background_images",
        'mnist_background_images_train.amat',
        'mnist_background_images_test.amat')
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "mnist_bgimg.hdf5"), rescale=True)


def _create_mnist_bgrand(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory,
        "mnist_background_random",
        'mnist_background_random_train.amat',
        'mnist_background_random_test.amat')
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "mnist_bgrand.hdf5"), rescale=True)


def _create_mnist_rot(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory,
        "mnist_rotation_new",
        'mnist_all_rotation_normalized_float_train_valid.amat',
        'mnist_all_rotation_normalized_float_test.amat')
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "mnist_rot.hdf5"), rescale=True)


def _create_rectangles(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory,
        "rectangles",
        'rectangles_train.amat',
        'rectangles_test.amat')
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "rectangles.hdf5"), rescale=True)


def _create_convex(directory):
    trainx, trainy, testx, testy = _handle_larochelle_icml2007(directory,
        "convex",
        'convex_train.amat',
        '50k/convex_test.amat')
    trainx, trainy, validx, validy = _split_dataset(trainx, trainy, 5/6.0)
    data = [['train', trainx, trainy],
            ['valid', validx, validy],
            ['test', testx, testy]]
    _process_and_store(data, os.path.join(directory, "convex.hdf5"), rescale=True)


def _create_covertype(directory):
    urlbase = 'https://archive.ics.uci.edu/ml/machine-learning-databases/covtype/'
    destdir = os.path.join(directory, "raw")
    fn = _download_file(urlbase, 'covtype.data.gz', destdir)
    with gzip.open(fn, "rb") as gzfile:
        X = pd.read_csv(gzfile, header=None).values

    X, y = X[:, :-1].astype(np.float64), X[:, -1]
    y -= 1 # make classes 0-based

    # split into test- and validationset
    idx = range(X.shape[0])
    from sklearn.cross_validation import train_test_split
    X, Xtest, y, ytest = train_test_split(X, y, test_size=0.1)
    X, Xval, y, yval = train_test_split(X, y, test_size=0.25)

    from sklearn.preprocessing import LabelBinarizer
    lb = LabelBinarizer()
    y = lb.fit_transform(y)
    yval = lb.transform(yval)
    ytest = lb.transform(ytest)

    # Most values are binary, except for these, so let's standardize them
    quant_idx = [0, 1, 2, 3, 4, 5, 9]  # real numbers
    int_idx = [6, 7, 8]                # integers from [0, 255)
    from sklearn.preprocessing import StandardScaler as Scaler
    scaler = Scaler()
    X[:, quant_idx + int_idx] = scaler.fit_transform(X[:, quant_idx+int_idx])
    Xval[:, quant_idx + int_idx] = scaler.transform(Xval[:, quant_idx + int_idx])
    Xtest[:, quant_idx + int_idx] = scaler.transform(Xtest[:, quant_idx + int_idx])
    data = [['train', X,  y],
            ['valid', Xval,  yval],
            ['test', Xtest,  ytest]]
    m = np.zeros(X.shape[1])
    m[quant_idx+int_idx] = scaler.mean_
    s = np.ones(X.shape[1])
    s[quant_idx+int_idx] = scaler.std_
    other = {'center': m, "scale": s}
    _store(data, os.path.join(directory, "covertype.hdf5"), other)


def _create_enwik8(directory):
    '''Prepares the enwik8/hutter prize data: an extract from wikipedia.'''
    urlbase = 'http://mattmahoney.net/dc/'
    destdir = os.path.join(directory, "raw")
    fn = _download_file(urlbase, 'enwik8.zip', destdir)

    # we first read the text as UTF-8, and then map each present character
    # to a number, instead of using UTF-8 bytes directly
    with zipfile.ZipFile(fn, "r") as zf:
        with zf.open("enwik8") as z:
            text_train = z.read(96*10**6).decode("utf8")
            text_valid = z.read(2*10**6).decode("utf8")
            text_test = z.read(2*10**6).decode("utf8")
            assert(len(z.read()) == 0) # make sure we read everything


    # ignore "uncommon" characters.
    # In "Generating Sequences With Recurrent Neural Networks"
    # Alex Graves says that there are 205 distinct single-byte characters.
    # However the following will only yield 196. No idea where Alex
    # got the rest of them ?-)
    data_tr = np.array([ord(c) for c in text_train if ord(c) < 256], dtype=np.uint8)
    data_va = np.array([ord(c) for c in text_valid if ord(c) < 256], dtype=np.uint8)
    data_te = np.array([ord(c) for c in text_test if ord(c) < 256], dtype=np.uint8)
    cnt = pd.value_counts(data_tr)

    del(text_train, text_valid, text_test)
    import gc
    gc.collect()

    # remove characters with <=10 occourences (there are 16 of those)
    # (we use a lookup table, othewise it takes forever)
    count_loopup = np.zeros(256, np.int64)
    count_loopup[cnt.index.values] = cnt.values
    occ = count_loopup[data_tr]
    data_tr = data_tr[occ > 10]
    data_va = data_va[count_loopup[data_va] > 10]
    data_te = data_te[count_loopup[data_te] > 10]

    decode_lookup = 255 * np.ones(256, np.uint8)
    u = np.unique(data_tr)
    decode_lookup[:len(u)] = u
    encode_lookup = np.iinfo(np.uint16).max * np.ones(256, np.uint16)
    for c, e in enumerate(u):
        encode_lookup[e] = c
    code_tr = encode_lookup[data_tr]
    code_va = encode_lookup[data_va]
    code_te = encode_lookup[data_te]
    assert(np.all(decode_lookup[code_tr] == data_tr))
    assert(np.all(code_tr <= 255))
    assert(np.all(code_va <= 255))
    assert(np.all(code_te <= 255))
    del(data_tr, data_va, data_te)
    gc.collect()

    fname = os.path.join(directory, "enwik8.hdf5")
    with h5py.File(fname, "w") as f:
        f.create_dataset('train', data=code_tr)
        f.create_dataset('valid', data=code_va)
        f.create_dataset('test', data=code_te)
        f.create_dataset('encode', data=encode_lookup)
        f.create_dataset('decode', data=decode_lookup)


def _create_tox21_impl(sparsity_cutoff, validation_fold, directory=_DATA_DIRECTORY):
    urlbase = "http://www.bioinf.jku.at/research/deeptox/"
    dst = os.path.join(directory, "raw")
    fn_x_tr_d = _download_file(urlbase, 'tox21_dense_train.csv.gz', dst)
    fn_x_tr_s = _download_file(urlbase, 'tox21_sparse_train.mtx.gz', dst)
    fn_y_tr = _download_file(urlbase, 'tox21_labels_train.csv', dst)
    fn_x_te_d = _download_file(urlbase, 'tox21_dense_test.csv.gz', dst)
    fn_x_te_s = _download_file(urlbase, 'tox21_sparse_test.mtx.gz', dst)
    fn_y_te = _download_file(urlbase, 'tox21_labels_test.csv', dst)
    cpd = _download_file(urlbase, 'tox21_compoundData.csv', dst)

    y_tr = pd.read_csv(fn_y_tr, index_col=0)
    y_te = pd.read_csv(fn_y_te, index_col=0)
    x_tr_dense = pd.read_csv(fn_x_tr_d, index_col=0).values
    x_te_dense = pd.read_csv(fn_x_te_d, index_col=0).values
    x_tr_sparse = io.mmread(fn_x_tr_s).tocsc()
    x_te_sparse = io.mmread(fn_x_te_s).tocsc()

    # filter out very sparse features
    sparse_col_idx = ((x_tr_sparse > 0).mean(0) >= sparsity_cutoff).A.ravel()
    x_tr_sparse = x_tr_sparse[:, sparse_col_idx].A
    x_te_sparse = x_te_sparse[:, sparse_col_idx].A

    dense_col_idx = np.where(x_tr_dense.var(0) > 1e-6)[0]
    x_tr_dense = x_tr_dense[:, dense_col_idx]
    x_te_dense = x_te_dense[:, dense_col_idx]

    # The validation set consists of those samples with
    # cross validation fold #5
    info = pd.read_csv(cpd, index_col=0)
    f = info.CVfold[info.set != 'test'].values
    idx_va = f == float(validation_fold)

    # normalize features
    from sklearn.preprocessing import StandardScaler
    s = StandardScaler()
    s.fit(x_tr_dense[~idx_va])
    x_tr_dense = s.transform(x_tr_dense)
    x_te_dense = s.transform(x_te_dense)

    x_tr_sparse = np.tanh(x_tr_sparse)
    x_te_sparse = np.tanh(x_te_sparse)

    x_tr = np.hstack([x_tr_dense, x_tr_sparse])
    x_te = np.hstack([x_te_dense, x_te_sparse])
    return (x_tr[~idx_va],  y_tr[~idx_va],
            x_tr[idx_va], y_tr[idx_va],
            x_te,  y_te)


def _create_tox21(directory):
    sparsity_cutoff = 0.05
    validation_fold = 5
    d = _create_tox21_impl(sparsity_cutoff, validation_fold)
    x_tr, y_tr, x_va, y_va, x_te, y_te = d
    data = [['train', x_tr,  y_tr],
            ['valid', x_va,  y_va],
            ['test',  x_te,  y_te]]
    _store(data, os.path.join(directory, "tox21.hdf5"))


if __name__ == "__main__":
    import sys
    directory = "./" if len(sys.argv) <= 2 else sys.argv[2]

    if len(sys.argv) < 2:
        print(__doc__)
    elif sys.argv[1].lower() == "mnist":
        _create_mnist(directory)
    elif sys.argv[1].lower() == "norb":
        _create_norb(directory)
    elif sys.argv[1].lower() == "norbhalf":
        _create_norb_half(directory)
    elif sys.argv[1].lower() == "cifar10":
        _create_cifar10(directory)
    elif sys.argv[1].lower() == "norb-downsampled":
        _create_norb_downsampled(directory)
    else:
        print("ERROR: unknown dataset.")
        print(__doc__)
