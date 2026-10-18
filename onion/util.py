import typing

import numpy as np

def one_hot(y: np.ndarray[np.int32], n_classes: int | None = None) -> np.ndarray[bool]:
    if n_classes is None:
        n_classes = np.max(y, axis=None)
    return np.eye(n_classes)[y]

def sigmoid(a: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return np.tanh(a)

def relu(a: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    return np.maximum(a, 0.0)

# numerically stable modification - credit to a friend of mine
def softmax(a: np.ndarray[np.float64]) -> np.ndarray[np.float64]:
    exp_a = np.exp(a - np.max(a, axis=-1, keepdims=True))
    return exp_a / np.sum(exp_a, axis=-1, keepdims=True)

def add_grads(*grads: dict[str, np.ndarray[np.float64]]) -> None:
    result = {name: np.zeros_like(arr) for grad in grads for name, arr in grad.items()}
    for grad in grads:
        for name, val in grad.items():
            result[name] += val
    return result

# loop over many hparams at once
def enum_hparams(
    hparams: dict[str, list[object]],
) -> typing.Iterable[dict[str, object]]:
    names = list(hparams.keys())
    for inds in np.ndindex(*[len(hparams[name]) for name in names]):
        yield {name: hparams[name][ind] for name, ind in zip(names, inds)}

# shuffle multiple arrays in parallel
def shuffle_arrays(
    *arrays: np.ndarray[np.float64],
    rng: np.random.Generator | int = 18,
) -> list[np.ndarray[np.float64]]:
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)
    perm = rng.permutation(len(arrays[0]))
    return [arr[perm] for arr in arrays]

# split X and y into two groups - train and test, or to only use a small fraction of the data
def train_test_split(
    *arrays: np.ndarray[np.float64],
    ratios: float | list[float] = 0.8,
    rng: np.random.Generator | int = 18,
    shuffle: bool = True,
) -> list[list[np.ndarray[np.float64]]]:
    if shuffle:
        arrays = shuffle_arrays(*arrays, rng=rng)

    if isinstance(ratios, float):
        ratios = [ratios]

    ratios = [0.0] + list(ratios)

    if sum(ratios) < 1.0:
        ratios.append(1.0 - sum(ratios))

    inds = (np.cumsum(ratios, dtype=np.float64) * len(arrays[0])).astype(np.int32)

    return [[arr[i:j] for (i, j) in zip(inds[:-1], inds[1:])] for arr in arrays]

def batch(
    *arrays: np.ndarray[np.float64],
    batch_size: int,
    rng: np.random.Generator | int = 18,
    shuffle: bool = True,
) -> list[list[np.ndarray[np.float64]]]:
    return train_test_split(
        *arrays,
        ratios=[batch_size / len(arrays[0])] * (len(arrays[0]) // batch_size),
        rng=rng,
        shuffle=shuffle
    )

def k_fold_split(
    *arrays: np.ndarray[np.float64],
    k: int = 4,
    rng: np.random.Generator | int = 18,
    shuffle: bool = True,
) -> tuple[
    list[list[np.ndarray[np.float64]]],
    list[list[np.ndarray[np.float64]]],
]:
    n = len(arrays[0])
    if shuffle:
        arrays = shuffle_arrays(*arrays, rng=rng)

    arrays = [np.repeat(arr, 2, axis=0) for arr in arrays]  # do a thing sorta like a circular array

    inds = np.linspace(0, n, k + 1).astype(np.int32)

    train, val = [[]] * len(arrays), [[]] * len(arrays)

    for start, stop in zip(inds[:-1], inds[1:]):
        for arr, trs, vals in zip(arrays, train, val):
            trs.append(arr[stop: start + n])
            vals.append(arr[start: stop])

    return train, val

class tr_:
    """
    tuple-range.
    index it with slice notation and it returns a tuple with the contents of the corresponding range object.
    useful for indexing shenaniganery.
    """
    @classmethod
    def __class_getitem__(cls, idx: int | slice) -> tuple:
        if isinstance(idx, int):
            idx = slice(idx)
        if idx.step is None:
            r = range(idx.stop) if idx.start is None else range(idx.start, idx.stop)
        else:
            r = range(0, idx.stop, idx.step) if idx.start is None else range(idx.start, idx.stop, idx.step)
        return tuple(r)

def stride_slices(st: int | list[int], ndim: int | None = None) -> tuple[slice]:
    if isinstance(st, int):
        st = (st,) * ndim
    return tuple(slice(None, None, s) for s in st)

def pad_arr(a: np.ndarray[np.float64], axis: int | list[int] | None = None, pad: int | tuple[int] = 1) -> np.ndarray[np.float64]:
    if axis is None:
        axis = tr_[:len(a.shape)]
    elif isinstance(axis, int):
        axis = axis,
    if isinstance(pad, int):
        pad = (pad,) * len(axis)
    pad_widths = [(pad[axis.index(i)] if i in axis else 0,) * 2 for i in range(len(a.shape))]
    return np.pad(a, pad_widths)

def grid(a: np.ndarray[np.float64], *tgt_shape: tuple[int]) -> np.ndarray[np.float64]:
    shape = sum([[d] if s == 0 else [d // s, s] for d, s in zip(a.shape, tgt_shape)], [])
    ax_a, ax_b, pos = [], [], 0
    for s in tgt_shape:
        ax_a.append(pos)
        pos += 1
        if s > 0:
            ax_b.append(pos)
            pos += 1
    return a.reshape(shape).transpose(ax_a + ax_b)