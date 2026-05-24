from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


TARGET_MODE_DISPLAY_NAMES = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}

FINAL_METHOD_DISPLAY_NAMES = {
    "heuristic_update": "Heuristic update",
    "raw_preference_model": "Raw Preference Model",
    "norm_calibrated_model": "Norm-calibrated PM",
    "train_scale_model": "Train-scale PM",
    "blend_70h_30m": "Selected: Blend 70/30",
    "blend_50h_50m": "Blend 50/50",
}

FINAL_METHOD_ORDER = [
    "heuristic_update",
    "raw_preference_model",
    "norm_calibrated_model",
    "train_scale_model",
    "blend_70h_30m",
    "blend_50h_50m",
]

ARTICLE_METHODS = [
    "heuristic_update",
    "norm_calibrated_model",
    "blend_70h_30m",
    "blend_50h_50m",
]

SELECTED_METHOD = "blend_70h_30m"
ARCHETYPE_BEST_METHOD = "norm_calibrated_model"


def ensure_dir(path: str | Path) -> Path:
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def add_display_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "target_mode" in df.columns:
        df["target_mode_display"] = df["target_mode"].map(TARGET_MODE_DISPLAY_NAMES).fillna(df["target_mode"])
    if "method" in df.columns:
        df["method_display"] = df["method"].map(FINAL_METHOD_DISPLAY_NAMES).fillna(df["method"])
    return df


def summarize_final_architecture(sessions: pd.DataFrame) -> pd.DataFrame:
    """Aggregate final-vector methods by target mode."""
    df = add_display_columns(sessions)
    summary = (
        df.groupby(["target_mode", "target_mode_display", "method", "method_display"])
        .agg(
            users=("user_id", "nunique"),
            mean_final_distance=("final_distance", "mean"),
            std_final_distance=("final_distance", "std"),
            mean_cosine_to_target=("cosine_to_target", "mean"),
            mean_heldout_accuracy=("heldout_accuracy", "mean"),
            mean_heldout_log_loss=("heldout_log_loss", "mean"),
            mean_vector_norm=("vector_norm", "mean"),
        )
        .reset_index()
    )
    return summary


def compare_methods_vs_heuristic(sessions: pd.DataFrame) -> pd.DataFrame:
    """Per-user improvement of final-vector methods against heuristic update."""
    df = add_display_columns(sessions)
    index_cols = ["user_id", "target_mode", "target_mode_display"]
    if "intensity_label" in df.columns:
        index_cols.append("intensity_label")
    if "main_archetype" in df.columns:
        index_cols.append("main_archetype")

    pivot = df.pivot_table(
        index=index_cols,
        columns="method",
        values="final_distance",
        aggfunc="mean",
    ).reset_index()

    rows: list[dict] = []
    if "heuristic_update" not in pivot.columns:
        return pd.DataFrame(rows)

    for method in FINAL_METHOD_ORDER:
        if method == "heuristic_update" or method not in pivot.columns:
            continue
        valid = pivot[["heuristic_update", method]].dropna()
        for _, row in pivot.dropna(subset=["heuristic_update", method]).iterrows():
            heuristic = float(row["heuristic_update"])
            value = float(row[method])
            improvement_abs = heuristic - value
            improvement_pct = 100.0 * improvement_abs / max(heuristic, 1e-8)
            out = {col: row[col] for col in index_cols}
            out.update({
                "method": method,
                "method_display": FINAL_METHOD_DISPLAY_NAMES.get(method, method),
                "heuristic_final_distance": heuristic,
                "method_final_distance": value,
                "improvement_abs": improvement_abs,
                "improvement_pct": improvement_pct,
                "method_wins": bool(value < heuristic),
            })
            rows.append(out)

    return pd.DataFrame(rows)


def summarize_method_improvements(improvement_df: pd.DataFrame) -> pd.DataFrame:
    if improvement_df.empty:
        return improvement_df
    return (
        improvement_df.groupby(["target_mode", "target_mode_display", "method", "method_display"])
        .agg(
            users=("user_id", "nunique"),
            mean_heuristic_final_distance=("heuristic_final_distance", "mean"),
            mean_method_final_distance=("method_final_distance", "mean"),
            mean_improvement_abs=("improvement_abs", "mean"),
            mean_improvement_pct=("improvement_pct", "mean"),
            win_rate_vs_heuristic=("method_wins", "mean"),
        )
        .reset_index()
    )


def final_architecture_description_table() -> pd.DataFrame:
    rows = [
        {
            "block": "Pair Generator",
            "selected_component": "Semantic active v3",
            "role": "Generates informative semantic A/B questions using 6D semantic basis and current uncertainty.",
        },
        {
            "block": "Online preference state",
            "selected_component": "Heuristic update",
            "role": "Maintains stable z_mean trajectory during the A/B session.",
        },
        {
            "block": "Parallel Preference Model",
            "selected_component": "Online Logistic Preference Model",
            "role": "Learns preference direction from pairwise A/B choices without controlling pair generation.",
        },
        {
            "block": "Final vector refinement",
            "selected_component": "Selected: Blend 70/30",
            "role": "Combines stable heuristic state with calibrated Preference Model direction.",
        },
        {
            "block": "Best Archetype 8D ablation",
            "selected_component": "Norm-calibrated PM",
            "role": "Uses the Preference Model direction with heuristic vector norm; strongest on realistic Archetype 8D in the current experiment.",
        },
    ]
    return pd.DataFrame(rows)


def _ordered_subset(df: pd.DataFrame, methods: Iterable[str] | None = None) -> pd.DataFrame:
    methods = list(methods or FINAL_METHOD_ORDER)
    out = df[df["method"].isin(methods)].copy()
    out["method"] = pd.Categorical(out["method"], categories=methods, ordered=True)
    target_order = ["random8d", "semantic4d", "semantic6d", "archetype8d"]
    out["target_mode"] = pd.Categorical(out["target_mode"], categories=target_order, ordered=True)
    return out.sort_values(["target_mode", "method"])


def plot_final_architecture_diagram(save_path: str | Path | None = None) -> None:
    """Draw a compact architecture diagram for the selected contour."""
    fig, ax = plt.subplots(figsize=(14, 4.8), facecolor="white")
    ax.axis("off")

    boxes = [
        (0.04, 0.55, 0.13, 0.24, "User /\nSynthetic user"),
        (0.22, 0.55, 0.15, 0.24, "A/B session\nchoice"),
        (0.42, 0.55, 0.18, 0.24, "Pair Generator\nSemantic active v3"),
        (0.66, 0.67, 0.16, 0.20, "Heuristic\nstate update"),
        (0.66, 0.38, 0.16, 0.20, "Parallel Logistic\nPreference Model"),
        (0.86, 0.55, 0.12, 0.24, "Final vector\ncalibration / blend"),
    ]

    for x, y, w, h, text in boxes:
        patch = FancyBboxPatch(
            (x, y), w, h,
            boxstyle="round,pad=0.018,rounding_size=0.025",
            linewidth=1.3,
            edgecolor="0.25",
            facecolor="white",
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=11)

    arrows = [
        ((0.17, 0.67), (0.22, 0.67)),
        ((0.37, 0.67), (0.42, 0.67)),
        ((0.60, 0.67), (0.66, 0.77)),
        ((0.60, 0.67), (0.66, 0.48)),
        ((0.82, 0.77), (0.86, 0.67)),
        ((0.82, 0.48), (0.86, 0.67)),
    ]
    for start, end in arrows:
        ax.annotate("", xy=end, xytext=start, arrowprops={"arrowstyle": "->", "lw": 1.4, "color": "0.2"})

    ax.text(0.50, 0.18, "Selected contour: Semantic active v3 + heuristic online state + calibrated Preference Model final vector", ha="center", fontsize=13, fontweight="bold")
    ax.text(0.50, 0.10, "Preference Model does not select A/B questions in this architecture; it refines the final preference vector after observing A/B history.", ha="center", fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_final_distance_by_target(summary: pd.DataFrame, save_path: str | Path | None = None) -> None:
    df = _ordered_subset(add_display_columns(summary), ARTICLE_METHODS)
    table = df.pivot(index="target_mode_display", columns="method_display", values="mean_final_distance")
    # Preserve target order after pivot.
    target_order = [TARGET_MODE_DISPLAY_NAMES[x] for x in ["random8d", "semantic4d", "semantic6d", "archetype8d"] if TARGET_MODE_DISPLAY_NAMES[x] in table.index]
    method_order = [FINAL_METHOD_DISPLAY_NAMES[m] for m in ARTICLE_METHODS if FINAL_METHOD_DISPLAY_NAMES[m] in table.columns]
    table = table.loc[target_order, method_order]

    fig, ax = plt.subplots(figsize=(13, 6), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Итоговая архитектура: mean final distance по режимам target", fontsize=16, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Final preference vector", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_archetype_final_vector_comparison(summary: pd.DataFrame, save_path: str | Path | None = None) -> None:
    df = _ordered_subset(add_display_columns(summary), FINAL_METHOD_ORDER)
    df = df[df["target_mode"] == "archetype8d"].copy()

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    labels = df["method_display"].tolist()
    values = df["mean_final_distance"].to_numpy(dtype=np.float64)
    ax.bar(labels, values)
    ax.set_title("Final vector methods: Archetype 8D", fontsize=16, fontweight="bold")
    ax.set_xlabel("Final preference vector")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    plt.xticks(rotation=25, ha="right")
    for i, val in enumerate(values):
        ax.text(i, val, f"{val:.3f}", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_selected_vs_heuristic_improvement(improvement_summary: pd.DataFrame, save_path: str | Path | None = None) -> None:
    df = add_display_columns(improvement_summary)
    df = df[df["method"].isin([SELECTED_METHOD, ARCHETYPE_BEST_METHOD])].copy()
    target_order = ["random8d", "semantic4d", "semantic6d", "archetype8d"]
    df["target_mode"] = pd.Categorical(df["target_mode"], categories=target_order, ordered=True)
    df = df.sort_values(["target_mode", "method"])
    table = df.pivot(index="target_mode_display", columns="method_display", values="mean_improvement_pct")
    target_labels = [TARGET_MODE_DISPLAY_NAMES[x] for x in target_order if TARGET_MODE_DISPLAY_NAMES[x] in table.index]
    method_labels = [FINAL_METHOD_DISPLAY_NAMES[m] for m in [ARCHETYPE_BEST_METHOD, SELECTED_METHOD] if FINAL_METHOD_DISPLAY_NAMES[m] in table.columns]
    table = table.loc[target_labels, method_labels]

    fig, ax = plt.subplots(figsize=(12, 5), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.7)
    ax.axhline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Улучшение финального вектора относительно heuristic update", fontsize=16, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Улучшение mean final distance, %")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Final preference vector", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_heldout_accuracy_by_target(summary: pd.DataFrame, save_path: str | Path | None = None) -> None:
    df = _ordered_subset(add_display_columns(summary), ARTICLE_METHODS)
    table = df.pivot(index="target_mode_display", columns="method_display", values="mean_heldout_accuracy")
    target_order = [TARGET_MODE_DISPLAY_NAMES[x] for x in ["random8d", "semantic4d", "semantic6d", "archetype8d"] if TARGET_MODE_DISPLAY_NAMES[x] in table.index]
    method_order = [FINAL_METHOD_DISPLAY_NAMES[m] for m in ARTICLE_METHODS if FINAL_METHOD_DISPLAY_NAMES[m] in table.columns]
    table = table.loc[target_order, method_order]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.82)
    ax.axhline(0.5, linestyle="--", linewidth=1.2, label="Random guess")
    ax.set_title("Held-out accuracy финального preference vector", fontsize=16, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Mean held-out accuracy")
    ax.set_ylim(0.0, 1.0)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Final preference vector", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def plot_cosine_by_target(summary: pd.DataFrame, save_path: str | Path | None = None) -> None:
    df = _ordered_subset(add_display_columns(summary), ARTICLE_METHODS)
    table = df.pivot(index="target_mode_display", columns="method_display", values="mean_cosine_to_target")
    target_order = [TARGET_MODE_DISPLAY_NAMES[x] for x in ["random8d", "semantic4d", "semantic6d", "archetype8d"] if TARGET_MODE_DISPLAY_NAMES[x] in table.index]
    method_order = [FINAL_METHOD_DISPLAY_NAMES[m] for m in ARTICLE_METHODS if FINAL_METHOD_DISPLAY_NAMES[m] in table.columns]
    table = table.loc[target_order, method_order]

    fig, ax = plt.subplots(figsize=(13, 5), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Близость направления финального вектора к скрытому target", fontsize=16, fontweight="bold")
    ax.set_xlabel("Режим генерации target")
    ax.set_ylabel("Mean cosine similarity")
    ax.set_ylim(0.0, 1.05)
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Final preference vector", frameon=True, facecolor="white", edgecolor="0.75")
    plt.xticks(rotation=0)
    plt.tight_layout()
    if save_path is not None:
        plt.savefig(save_path, dpi=220, bbox_inches="tight")
    plt.show()


def save_final_architecture_tables(
    sessions: pd.DataFrame,
    heldout_pairs: pd.DataFrame,
    train_steps: pd.DataFrame,
    summary: pd.DataFrame,
    improvement: pd.DataFrame,
    improvement_summary: pd.DataFrame,
    output_dir: str | Path,
    prefix: str = "article_final_architecture",
) -> None:
    output_dir = ensure_dir(output_dir)
    sessions.to_csv(output_dir / f"{prefix}_sessions.csv", index=False)
    heldout_pairs.to_csv(output_dir / f"{prefix}_heldout_pairs.csv", index=False)
    train_steps.to_csv(output_dir / f"{prefix}_train_steps.csv", index=False)
    summary.to_csv(output_dir / f"{prefix}_summary.csv", index=False)
    improvement.to_csv(output_dir / f"{prefix}_improvement_vs_heuristic.csv", index=False)
    improvement_summary.to_csv(output_dir / f"{prefix}_improvement_summary.csv", index=False)
    final_architecture_description_table().to_csv(output_dir / f"{prefix}_description.csv", index=False)
