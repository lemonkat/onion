"""
Onion: A custom ML library.
It's called that because it has layers.
And definitiely not becuase it's an absolute joke.
"""

from onion.util import (
    train_test_split,
    batch,
    k_fold_split,
    one_hot,
    relu,
    softmax,
    add_grads,
    tr_,
    stride_slices,
    pad_arr,
    grid,
)
from onion.layer import (
    Layer,
    Parameter,
    Identity,
    Add,
    Dense,
    ReLU,
    Sigmoid,
    L2Loss,
    CrossEntropyLoss,
    L2Regularization,
    Dropout,
    BatchNorm,
    Convolution,
    Reshape,
    MaxPool,
)
from onion.model import Model, DAGModel, DAGNode, node
from onion.optimizer import Optimizer, SGD, ADAM
from onion.training import train, hparam_search, train_async, metrics_async
from onion.plot import plot_hist

from onion import util, layer, model, optimizer, training, plot

__all__ = [
    "util",
    "layer",
    "model",
    "optimizer",
    "training",
    "plot",
    "train_test_split",
    "batch",
    "k_fold_split",
    "one_hot",
    "relu",
    "tr_",
    "pad_arr",
    "grid",
    "softmax",
    "add_grads",
    "stride_slices",
    "train",
    "train_async",
    "metrics_async",
    "hparam_search",
    "plot_hist",
    "Layer",
    "Parameter",
    "Model",
    "Optimizer",
    "Identity",
    "Add",
    "Dense",
    "ReLU",
    "Sigmoid",
    "L2Loss",
    "CrossEntropyLoss",
    "L2Regularization",
    "Dropout",
    "BatchNorm",
    "Convolution",
    "Reshape",
    "MaxPool",
    "SGD",
    "ADAM",
    "DAGModel",
    "DAGNode",
    "node",
]
