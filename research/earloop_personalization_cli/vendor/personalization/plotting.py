from __future__ import annotations

import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

from .state import FEATURE_NAMES_8D


ARTICLE_RC_PARAMS = {
    "figure.facecolor": "white",
    "figure.edgecolor": "white",
    "axes.facecolor": "white",
    "axes.edgecolor": "black",
    "axes.labelcolor": "black",
    "axes.titlecolor": "black",
    "xtick.color": "black",
    "ytick.color": "black",
    "text.color": "black",
    "legend.facecolor": "white",
    "legend.edgecolor": "0.75",
    "legend.framealpha": 0.95,
    "savefig.facecolor": "white",
    "savefig.edgecolor": "white",
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
}


def use_article_style() -> None:
    """
    Apply a light publication-style matplotlib theme.

    This is useful in PyCharm/Jupyter when the IDE uses a dark plotting theme.
    Call it once before plotting, or rely on the plotting functions below,
    which call it automatically.
    """
    mpl.rcParams.update(ARTICLE_RC_PARAMS)


def _style_article_axes(ax, title: str | None = None) -> None:
    ax.set_facecolor("white")

    if title is not None:
        ax.set_title(title, fontsize=15, fontweight="bold", color="black", pad=12)

    ax.grid(True, which="major", linestyle="--", linewidth=0.8, alpha=0.35, color="gray")
    ax.grid(True, which="minor", linestyle="--", linewidth=0.5, alpha=0.15, color="gray")

    ax.tick_params(axis="both", colors="black", labelsize=10)
    ax.xaxis.label.set_color("black")
    ax.yaxis.label.set_color("black")

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)


def plot_convergence(
    distances: np.ndarray,
    title: str = "Сходимость personalization loop v0",
):
    """
    Plot distance from current preference state to hidden user target.
    """
    use_article_style()

    distances = np.asarray(distances, dtype=float)
    steps = np.arange(1, len(distances) + 1)

    fig, ax = plt.subplots(figsize=(9, 4.8), facecolor="white")
    fig.patch.set_facecolor("white")

    ax.plot(
        steps,
        distances,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="distance to target",
    )

    ax.set_xlabel("Шаг A/B-сессии", color="black", labelpad=8)
    ax.set_ylabel("Расстояние до скрытого target", color="black", labelpad=8)
    _style_article_axes(ax, title=title)

    ax.set_xlim(1, len(distances))
    y_min = max(0.0, float(distances.min()) - 0.05)
    y_max = float(distances.max()) + 0.08
    ax.set_ylim(y_min, y_max)

    leg = ax.legend(loc="best", frameon=True)
    for text in leg.get_texts():
        text.set_color("black")

    plt.tight_layout()
    return fig, ax


def plot_final_vs_target(
    z_final: np.ndarray,
    z_target: np.ndarray,
    feature_names: list[str] | None = None,
    title: str = "Скрытый target и финальный preference state",
):
    """
    Compare hidden user target with the final estimated preference state.
    """
    use_article_style()

    names = FEATURE_NAMES_8D if feature_names is None else feature_names
    x = np.arange(len(names))

    z_final = np.asarray(z_final, dtype=float)
    z_target = np.asarray(z_target, dtype=float)

    fig, ax = plt.subplots(figsize=(10.5, 5.0), facecolor="white")
    fig.patch.set_facecolor("white")

    ax.plot(
        x,
        z_target,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="target",
    )
    ax.plot(
        x,
        z_final,
        marker="o",
        linewidth=2.2,
        markersize=5,
        label="final",
    )

    ax.set_xticks(x)
    ax.set_xticklabels(names, rotation=30, ha="right", color="black")
    ax.set_ylabel("Значение compact-признака", color="black", labelpad=8)
    _style_article_axes(ax, title=title)

    y_all = np.concatenate([z_target, z_final])
    y_min = float(y_all.min()) - 0.12
    y_max = float(y_all.max()) + 0.12
    ax.set_ylim(y_min, y_max)

    leg = ax.legend(loc="best", frameon=True)
    for text in leg.get_texts():
        text.set_color("black")

    plt.tight_layout()
    return fig, ax


def save_figure(fig, path: str, dpi: int = 300) -> None:
    """
    Save a figure in a thesis/article-friendly format with white background.
    """
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor="white", edgecolor="white")



def plot_semantic_control_basis_heatmap(
    basis_4d_to_8d: np.ndarray,
    control_names: list[str],
    feature_names: list[str],
    title: str = "Semantic basis: переход в weighted 8D",
):
    """Plot a heatmap showing how semantic controls map into weighted 8D."""
    use_article_style()

    basis = np.asarray(basis_4d_to_8d, dtype=float)

    fig, ax = plt.subplots(figsize=(10, 4.8), facecolor="white")
    im = ax.imshow(basis, aspect="auto")

    ax.set_xticks(np.arange(len(feature_names)))
    ax.set_xticklabels(feature_names, rotation=0, ha="center", color="black")
    ax.set_yticks(np.arange(len(control_names)))
    ax.set_yticklabels(control_names, color="black")
    ax.set_title(title, fontsize=15, color="black", pad=12)

    for i in range(basis.shape[0]):
        for j in range(basis.shape[1]):
            ax.text(
                j,
                i,
                f"{basis[i, j]:.2f}",
                ha="center",
                va="center",
                fontsize=8,
                color="black",
            )

    for spine in ax.spines.values():
        spine.set_color("black")
        spine.set_linewidth(1.0)

    fig.colorbar(im, ax=ax, shrink=0.8)
    plt.tight_layout()
    return fig, ax


def plot_semantic_control_influence(
    basis_4d_to_8d: np.ndarray,
    control_names: list[str],
    feature_names: list[str],
    title_prefix: str = "Влияние semantic control",
):
    """Plot one influence profile per semantic control over 8D features."""
    use_article_style()

    basis = np.asarray(basis_4d_to_8d, dtype=float)
    n_controls = basis.shape[0]
    x = np.arange(len(feature_names))

    fig, axes = plt.subplots(
        n_controls,
        1,
        figsize=(10, 2.45 * n_controls),
        sharex=True,
        facecolor="white",
    )
    if n_controls == 1:
        axes = [axes]

    for i, ax in enumerate(axes):
        weights = basis[i]

        ax.axhline(0.0, linewidth=1.0, alpha=0.7, color="black")
        ax.bar(x, weights, alpha=0.85)
        ax.plot(x, weights, marker="o", linewidth=1.6)

        ax.set_title(f"{title_prefix}: {control_names[i]}", fontsize=13, color="black", pad=8)
        ax.set_ylabel("Вес", color="black")
        ax.grid(True, axis="y", alpha=0.25, linestyle="--", color="gray")
        ax.tick_params(axis="both", colors="black")

        y_abs = max(0.1, float(np.max(np.abs(weights))))
        ax.set_ylim(-1.25 * y_abs, 1.25 * y_abs)

        for j, value in enumerate(weights):
            offset = 0.06 * y_abs if value >= 0 else -0.08 * y_abs
            ax.text(j, value + offset, f"{value:.2f}", ha="center", va="center", fontsize=8, color="black")

        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linewidth(1.0)

    axes[-1].set_xticks(x)
    axes[-1].set_xticklabels(feature_names, rotation=0, ha="center", color="black")
    axes[-1].set_xlabel("Признак weighted 8D", color="black")

    plt.tight_layout()
    return fig, axes


def plot_intensity_distribution(dataset, title: str = "Распределение synthetic users по интенсивности"):
    """Plot distribution of intensity labels for archetype8d users."""
    use_article_style()
    if "intensity_label" not in dataset.columns:
        raise ValueError("dataset must contain an 'intensity_label' column")

    counts = dataset["intensity_label"].value_counts().reindex(
        ["mild", "moderate", "strong", "extreme", "semantic", "random"],
        fill_value=0,
    )
    counts = counts[counts > 0]

    fig, ax = plt.subplots(figsize=(8, 4.5), facecolor="white")
    ax.bar(np.arange(len(counts)), counts.values)
    ax.set_xticks(np.arange(len(counts)))
    ax.set_xticklabels(counts.index, rotation=0, ha="center", color="black")
    ax.set_ylabel("Число пользователей", color="black")
    _style_article_axes(ax, title=title)

    for i, value in enumerate(counts.values):
        ax.text(i, value + max(counts.values) * 0.02, str(int(value)), ha="center", va="bottom", fontsize=10, color="black")

    plt.tight_layout()
    return fig, ax



def plot_average_convergence_by_strategy(
    curves_by_strategy: dict[str, np.ndarray],
    strategy_display_names: dict[str, str] | None = None,
    title: str = "Средняя сходимость A/B-сессии",
):
    """Plot mean distance-to-target curves for several strategies."""
    use_article_style()
    strategy_display_names = {} if strategy_display_names is None else strategy_display_names

    fig, ax = plt.subplots(figsize=(9, 5), facecolor="white")

    for strategy, curves in curves_by_strategy.items():
        curves = np.asarray(curves, dtype=float)
        if curves.size == 0:
            continue
        mean = curves.mean(axis=0)
        std = curves.std(axis=0)
        steps = np.arange(len(mean))
        label = strategy_display_names.get(strategy, strategy)
        ax.plot(steps, mean, marker="o", linewidth=2.0, label=label)
        ax.fill_between(steps, mean - std, mean + std, alpha=0.12)

    ax.set_title(title, fontsize=15, fontweight="bold", color="black", pad=12)
    ax.set_xlabel("Шаг A/B-сессии", color="black")
    ax.set_ylabel("Расстояние до скрытого target", color="black")
    ax.grid(True, alpha=0.3, linestyle="--", color="gray")
    ax.legend(title="Стратегия", frameon=True, facecolor="white", edgecolor="0.75")

    for spine in ax.spines.values():
        spine.set_color("black")

    plt.tight_layout()
    return fig, ax
