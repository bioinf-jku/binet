BINET
=====

binet is a Deep Learning library for Python that implements Deep Neural
Networks, and was developed at the
`Institute of Bioinformatics <http://www.bioinf.jku.at`_
at the `Johannes Kepler University Linz <http://www.jku.at>`_.

binet can seamlessly and transparently switch between running on the CPU and
on the GPU, using PyCUDA and scikits-cuda.

The library was written with the goal of easily experimenting with new
ideas regarding neural nets. While it is written with high performance
in mind, ease of extensibility and to internal net state was
a stated design goal.


Examples
--------

A simple neural network on MNIST with 2 hidden layers::



```python
    import os
    from binet import *

    op.init_gpu(0)   #  OPTIONAL: initializes first GPU in the system

    from binet.util import train
    dataset = load_dataset("mnist")

    layers = (dataset[0].shape[1], 256, 256, dataset[1].shape[1])
    net = NeuralNet(layers, max_iter=10, learning_rate=0.1, verbose=True,
                    activation="relu", shuffle_data=False, dropout=0.5,
                    input_dropout=0.2)
    net = train(net, dataset, use_gpu=True, skip_output=1)
```

Installation
------------
binet requires:

* numpy
* scipy
* pandas
* cffi
* Cython
* h5py (optionally, for ``load_dataset``)
* PyCUDA
* scikits.cuda
* GNU Scientific Library


License
-------
binet is licensed GPL, Version 2 or higher. It also uses some code
from the scikits.cuda, scikit-learn and cudamat project. See ``License.rst``
for more details.
