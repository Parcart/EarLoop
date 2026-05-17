"""Plotting helpers for mapper_v2 notebooks."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .feature_space import FEATURE_NAMES_8D


def use_article_style() -> None:
    """Use a light, notebook-safe style.

    Some IDE/Jupyter themes leave Matplotlib text colors inherited from a
    dark theme. In that case figures get a white background with white labels
    and the plot looks empty except for lines/markers. This style explicitly
    forces all text/ticks/legend colors to black.
    """
    plt.rcParams.update({
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.edgecolor": "white",
        "text.color": "black",
        "axes.labelcolor": "black",
        "axes.titlecolor": "black",
        "axes.edgecolor": "black",
        "xtick.color": "black",
        "ytick.color": "black",
        "legend.labelcolor": "black",
        "legend.facecolor": "white",
        "legend.edgecolor": "0.75",
        "axes.grid": True,
        "grid.alpha": 0.25,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "font.size": 11,
        "axes.titlesize": 15,
        "axes.labelsize": 12,
        "legend.frameon": True,
    })


def force_readable_figure(fig=None) -> None:
    """Force text and ticks to black on already-created figures.

    This is intentionally called by plotting helpers because notebooks may have
    loaded a dark Matplotlib style before importing this module.
    """
    fig = fig or plt.gcf()
    fig.patch.set_facecolor("white")

    for ax in fig.axes:
        ax.set_facecolor("white")
        ax.title.set_color("black")
        ax.xaxis.label.set_color("black")
        ax.yaxis.label.set_color("black")
        ax.tick_params(axis="both", colors="black", labelcolor="black")

        for spine in ax.spines.values():
            spine.set_color("black")

        for text in ax.get_xticklabels() + ax.get_yticklabels():
            text.set_color("black")

        legend = ax.get_legend()
        if legend is not None:
            legend.get_frame().set_facecolor("white")
            legend.get_frame().set_edgecolor("0.75")
            for text in legend.get_texts():
                text.set_color("black")

        # Colorbar axes are regular axes too, but their labels/ticks may be set
        # after creation. The generic handling above covers them.


def save_current_figure(path: str | Path, dpi: int = 160) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    force_readable_figure(plt.gcf())
    plt.tight_layout()
    plt.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white")


def format_freq_label(freq: float) -> str:
    freq = float(freq)
    if freq >= 1000:
        v = freq / 1000.0
        if abs(v - round(v)) < 1e-6:
            return f"{int(round(v))}k"
        return f"{v:g}k"
    return f"{freq:g}"


def style_freq_axis(ax, freqs: Iterable[float]) -> None:
    freqs = np.asarray(freqs, dtype=float)
    ax.set_xscale("log")
    ax.set_xticks(freqs)
    ax.set_xticklabels([format_freq_label(f) for f in freqs], rotation=45, ha="right")
    ax.set_xlabel("Frequency, Hz")


def plot_weight_bank(weight_bank: dict[str, np.ndarray], freqs: Iterable[float], title: str = "Weighted 8D extractor bank"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for name, w in weight_bank.items():
        ax.plot(freqs, w, marker="o", label=name)
    style_freq_axis(ax, freqs)
    ax.set_ylabel("Weight")
    ax.set_title(title)
    ax.legend(ncol=4)
    force_readable_figure(fig)
    plt.tight_layout()
    return fig, ax


def plot_basis_heatmap(basis_matrix: np.ndarray, freqs: Iterable[float], feature_names: list[str] | None = None, title: str = "8D → 23-band basis"):
    names = feature_names or FEATURE_NAMES_8D
    fig, ax = plt.subplots(figsize=(13, 5))
    im = ax.imshow(basis_matrix, aspect="auto", cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_yticks(np.arange(len(names)))
    ax.set_yticklabels(names)
    ax.set_xticks(np.arange(len(freqs)))
    ax.set_xticklabels([format_freq_label(f) for f in freqs], rotation=45, ha="right")
    ax.set_title(title)
    ax.set_xlabel("Frequency")
    fig.colorbar(im, ax=ax, label="Normalized template value")
    force_readable_figure(fig)
    plt.tight_layout()
    return fig, ax


def plot_eq_curves(curves: dict[str, np.ndarray], freqs: Iterable[float], title: str = "EQ curves", ylabel: str = "Gain, dB"):
    fig, ax = plt.subplots(figsize=(12, 5))
    for label, curve in curves.items():
        ax.plot(freqs, curve, marker="o", label=label)
    ax.axhline(0, linewidth=1, alpha=0.5)
    style_freq_axis(ax, freqs)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    ax.legend(ncol=2)
    force_readable_figure(fig)
    plt.tight_layout()
    return fig, ax


def plot_metric_bar(df: pd.DataFrame, x: str, y: str, title: str, ylabel: str | None = None, rotation: int = 25):
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(df[x].astype(str), df[y].astype(float))
    ax.set_title(title)
    ax.set_ylabel(ylabel or y)
    ax.set_xlabel(x)
    ax.tick_params(axis="x", rotation=rotation)
    force_readable_figure(fig)
    plt.tight_layout()
    return fig, ax
