import typing
import random
import os
import time

import numpy as np
from tqdm import tqdm

from onion.util import train_test_split, batch, k_fold_split, enum_hparams
from onion.model import Model
from onion.optimizer import Optimizer


def train(
    model: Model,
    opt: Optimizer,
    X_train: np.ndarray[np.float64],
    y_train: np.ndarray[np.int32],
    X_val: np.ndarray[np.float64] | None = None,
    y_val: np.ndarray[np.int32] | None = None,
    n_epochs: int = 1000,
    batch_size: int | None = None,
    rng: np.random.Generator | int = 18,
    metrics: dict[str, typing.Callable[[int, Model], object]] | None = None,
    include_default_metrics: bool = True,
    print_every: int = 1,
    prog_bar: bool = True,
    save_path: str | None = None,
) -> dict[str, list[object]]:
    
    model.eval()

    if save_path:
        model.save(save_path)
    
    if metrics is None:
        metrics = {}

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # default metrics
    if include_default_metrics:
        metrics |= {
            "Epoch": lambda epoch, model: epoch + 1,
            "Loss": lambda epoch, model: model.loss(X_train, y_train),
            "Acc": lambda epoch, model: model.accuracy(X_train, y_train),
        }

        if X_val is not None:
            metrics |= {
                "Val_Loss": lambda epoch, model: model.loss(X_val, y_val),
                "Val_Acc": lambda epoch, model: model.accuracy(X_val, y_val),
            }

    best_loss = float("inf")

    hist = {metric: [] for metric in metrics}

    metrics = [(name, metric) for name, metric in metrics.items()]

    for name, metric in metrics:
        if (value := metric(-1, model)) is not None:
            print(name + ":", value)

    for epoch in range(n_epochs):
        model.train()

        if batch_size is None:
            X_batch = [X_train]
            y_batch = [y_train]

        else:
            X_batch, y_batch = batch(
                X_train,
                y_train,
                batch_size=batch_size,
                rng=rng,
                shuffle=True,
            )
        if prog_bar:
            for X, y in tqdm(zip(X_batch, y_batch), total=len(X_batch)):
                opt.step(model.grad(X, y))
        else:
            for X, y in zip(X_batch, y_batch):
                opt.step(model.grad(X, y))

        model.eval()

        if X_val is not None:
            loss = model.loss(X_val, y_val)
        else:
            loss = model.loss(X_train, y_train)

        if metrics and print_every and (epoch + 1) % print_every == 0:
            print()

        for name, metric in metrics:
            value = metric(epoch, model)
            hist[name].append(value)
            if print_every > 0 and (epoch + 1) % print_every == 0:
                print(name + ":", value)

        if save_path and np.isfinite(loss) and loss < best_loss:
            model.save(save_path)
            best_loss = loss
            
            if print_every:
                print("New best!")

    if save_path:
        return model.load(save_path), hist
    return model, hist


def train_async(
    model: Model,
    opt: Optimizer,
    X: np.ndarray[np.float64],
    y: np.ndarray[np.int32],
    n_epochs: int = 1000,
    batch_size: int | None = None,
    rng: np.random.Generator | int = 18,
    prog_bar: bool = True,
    save_path: str | None = None,
) -> None:
    if save_path:
        model.save(save_path + "/epoch_0.onn")

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    model.train()

    for epoch in range(n_epochs):
        if prog_bar:
            print("Epoch", epoch)

        if batch_size is None:
            X_batch = [X]
            y_batch = [y]

        else:
            X_batch, y_batch = batch(
                X,
                y,
                batch_size=batch_size,
                rng=rng,
                shuffle=True,
            )

        if prog_bar:
            for Xb, yb in tqdm(zip(X_batch, y_batch), total=len(X_batch)):
                opt.step(model.grad(Xb, yb))
        else:
            for Xb, yb in zip(X_batch, y_batch):
                opt.step(model.grad(Xb, yb))


        model.save(save_path + f"/epoch_{epoch + 1}.onn")

    return model

def metrics_async(
    model_path: str,
    X_train: np.ndarray[np.float64],
    y_train: np.ndarray[np.int32],
    X_val: np.ndarray[np.float64] | None = None,
    y_val: np.ndarray[np.int32] | None = None,
    loader: typing.Callable[[str], Model] = Model.load,
    n_epochs: int = 1000,
    rng: np.random.Generator | int = 18,
    metrics: dict[str, typing.Callable[[int, Model], object]] | None = None,
    include_default_metrics: bool = True,
    save_metric: str | None = None,
    save_path: str | None = None,
    want_down: bool = True,
    prog_bar: bool = True,
) -> dict[str, list[object]]:
    if metrics is None:
        metrics = {}

    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    # default metrics
    if include_default_metrics:
        metrics |= {
            # "Loss": lambda model: model.loss(X_train, y_train),
            "Acc": lambda model: model.accuracy(X_train, y_train),
        }

        if X_val is not None:
            metrics |= {
                # "Val_Loss": lambda model: model.loss(X_val, y_val),
                "Val_Acc": lambda model: model.accuracy(X_val, y_val),
            }


    hist = {metric: [] for metric in metrics}

    if save_metric is not None:
        assert save_metric in metrics
        best_save_metric = float("inf") if want_down else float("-inf")

    metrics = [(name, metric) for name, metric in metrics.items()]

    for epoch in range(n_epochs + 1):
        path = model_path + f"/epoch_{epoch}.onn"
        while not os.path.exists(path):
            time.sleep(1.0)
        time.sleep(1.0) # time to finish writing
        model = loader(path)
        if prog_bar:
            model.eval()
        for name, metric in metrics:
            hist[name].append(metric(model))
            print(f"{name}: {hist[name][-1]}")
        if save_metric is not None and np.isfinite(hist[save_metric][-1]):
            if want_down == (hist[save_metric][-1] < best_save_metric):
                best_save_metric = hist[save_metric][-1]
                print("New best!")
                if save_path is not None:
                    model.save(save_path)
    return hist


def hparam_search(
    X_train_val: np.ndarray[np.float64],
    y_train_val: np.ndarray[np.int32],
    trial_fn: typing.Callable[
        [
            np.ndarray[np.float64],
            np.ndarray[np.int32],
            np.ndarray[np.float64],
            np.ndarray[np.int32],
            dict[str, object],
            np.random.Generator | int,
        ],
        tuple[Model, np.float64],
    ],
    hparam_opts: dict[str, list[object]],
    k: int | float = 4,
    rng: np.random.Generator | int = 18,

) -> dict[str, object]:
    if isinstance(rng, int):
        rng = np.random.default_rng(rng)

    if isinstance(k, int):
        X_train, X_val, y_train, y_val = k_fold_split(X_train_val, y_train_val, k, rng)
    
    else:
        (X_train, X_val), (y_train, y_val) = train_test_split(
            X_train_val, y_train_val, k, rng
        )
        X_train, X_val, y_train, y_val = [X_train], [X_val], [y_train], [y_val]
        k = 1

    fixed_hps = {key: val[0] for key, val in hparam_opts.items() if len(val) == 1}
    var_hp_opts = {key: val for key, val in hparam_opts.items() if len(val) != 1}

    hparams = list(enum_hparams(var_hp_opts))
    best_hps = None
    best_loss = float("inf")

    random.shuffle(hparams)

    with tqdm(total=k * len(hparams)) as bar:
        for var_hps in hparams:
            bar.set_description_str(",".join(f"{key}={val}" for key, val in var_hps.items()))
            loss = 0.0
            for i in range(k):
                loss += trial_fn(
                    X_train[i], y_train[i], X_val[i], y_val[i], fixed_hps | var_hps, rng
                )[1]
                bar.update(1)
            loss /= k
            if loss < best_loss:
                print("new best!")
                best_hps = var_hps
                best_loss = loss
    best_hps |= fixed_hps
    print(
        "Final parameters: "
        + ",".join(f"{key}={val}" for key, val in best_hps.items())
    )
    print(f"Final loss: {best_loss}")

    return best_hps