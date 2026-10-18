
import matplotlib.figure

import matplotlib
import matplotlib.pyplot as plt

def plot_hist(
    hist: dict[str, list[object]], 
    plot_groups: list[tuple[str | None, str | None, str | None, list[str]]] = [
        ("Loss", "Epoch", "Cross-Entropy Loss", ["Loss", "Val_Loss"]),
        ("Accuracy", "Epoch", "Accuracy", ["Acc", "Val_Acc"]),
    ],
    fig: matplotlib.figure.Figure | None = None,
) -> None:

    if fig is None:
        fig = plt.figure()

    axs = fig.subplots(1, len(plot_groups), sharex=True, sharey=False) if len(plot_groups) > 1 else [fig]

    for ax, (title, xlabel, ylabel, metrics) in zip(axs, plot_groups):
        if title:
            ax.set_title(title)
        if xlabel:
            ax.set_xlabel(xlabel)
        if ylabel:
            ax.set_ylabel(ylabel)
        for metric in metrics: 
            if metric in hist:
                ax.plot(hist[metric], label=metric)
        ax.legend()




