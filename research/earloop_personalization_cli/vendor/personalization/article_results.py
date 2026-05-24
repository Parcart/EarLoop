from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from .analysis import merge_sessions_with_user_metadata, summarize_by_group, win_rates_vs_baseline_by_group
from .batch_eval import run_batch_on_dataset, summarize_by_strategy, win_rates_vs_baseline
from .plotting import use_article_style, save_figure


ARTICLE_STRATEGIES = [
    "random",
    "uncertainty_axis",
    "semantic_control",
    "semantic_control_v21",
    "semantic_active_v21",
    "candidate_pool_active",
    "adaptive_router_v32",
]

ARTICLE_STRATEGY_LABELS = {
    "random": "Random direction",
    "uncertainty_axis": "Uncertainty axis",
    "semantic_control": "Semantic 4D",
    "semantic_control_v21": "Semantic 6D v2.1",
    "semantic_active_v21": "Semantic active v3",
    "candidate_pool_active": "Candidate pool active",
    "adaptive_router_v32": "Adaptive router v3.2",
}

TARGET_MODE_LABELS = {
    "random8d": "Random 8D",
    "semantic4d": "Semantic 4D",
    "semantic6d": "Semantic 6D",
    "archetype8d": "Archetype 8D",
}

INTENSITY_ORDER = ["mild", "moderate", "strong", "extreme"]
TARGET_MODE_ORDER = ["random8d", "semantic4d", "semantic6d", "archetype8d"]


STRATEGY_DESCRIPTIONS = pd.DataFrame([
    {
        "version": "V0",
        "strategy": "random",
        "display_name": ARTICLE_STRATEGY_LABELS["random"],
        "short_description": "Случайное многомерное направление в weighted 8D.",
        "role": "Базовый baseline для проверки, лучше ли новая стратегия случайного поиска.",
    },
    {
        "version": "V1",
        "strategy": "uncertainty_axis",
        "display_name": ARTICLE_STRATEGY_LABELS["uncertainty_axis"],
        "short_description": "A/B-вопрос по одной 8D-оси, выбранной по текущей неопределённости z_std.",
        "role": "Интерпретируемый baseline: уточнение конкретного compact-признака.",
    },
    {
        "version": "V2",
        "strategy": "semantic_control",
        "display_name": ARTICLE_STRATEGY_LABELS["semantic_control"],
        "short_description": "4D semantic basis: low_power, warmth_body, presence_clarity, air_brightness.",
        "role": "Проверка гипотезы о полезности музыкально осмысленных направлений.",
    },
    {
        "version": "V2.1",
        "strategy": "semantic_control_v21",
        "display_name": ARTICLE_STRATEGY_LABELS["semantic_control_v21"],
        "short_description": "6D semantic basis: V2 + club_energy + clean_bass.",
        "role": "Расширенная semantic basis для club/bass-oriented и экстремальных профилей.",
    },
    {
        "version": "V3",
        "strategy": "semantic_active_v21",
        "display_name": ARTICLE_STRATEGY_LABELS["semantic_active_v21"],
        "short_description": "Активный выбор semantic-вопроса по question-usefulness score.",
        "role": "Основной кандидат Pair Generator для реалистичных пользователей.",
    },
    {
        "version": "V3.1",
        "strategy": "candidate_pool_active",
        "display_name": ARTICLE_STRATEGY_LABELS["candidate_pool_active"],
        "short_description": "Смешанный пул random / axis / semantic кандидатов с active selection.",
        "role": "Более универсальный mixed-exploration baseline.",
    },
    {
        "version": "V3.2",
        "strategy": "adaptive_router_v32",
        "display_name": ARTICLE_STRATEGY_LABELS["adaptive_router_v32"],
        "short_description": "Эвристический router между semantic-active, candidate-pool и uncertainty-axis.",
        "role": "Экспериментальная попытка адаптивного переключения стратегий.",
    },
])


def _strategy_label(strategy: str) -> str:
    return ARTICLE_STRATEGY_LABELS.get(str(strategy), str(strategy))


def _target_label(mode: str) -> str:
    return TARGET_MODE_LABELS.get(str(mode), str(mode))


def _ordered_existing(values: Iterable[str], order: list[str]) -> list[str]:
    values = list(dict.fromkeys([str(v) for v in values]))
    return [v for v in order if v in values] + [v for v in values if v not in order]


def ensure_article_dirs(project_root: str | Path) -> dict[str, Path]:
    root = Path(project_root)
    dirs = {
        "metrics": root / "outputs" / "metrics",
        "figures": root / "outputs" / "figures",
        "tables": root / "outputs" / "tables",
    }
    for path in dirs.values():
        path.mkdir(parents=True, exist_ok=True)
    return dirs


def run_article_batch(
    dataset: pd.DataFrame,
    strategies: Iterable[str] = ARTICLE_STRATEGIES,
    n_steps: int = 25,
    step_scale: float = 0.6,
    lr: float = 0.25,
):
    """Run the fixed article comparison batch and return sessions, curves, summaries."""
    sessions, curves = run_batch_on_dataset(
        dataset=dataset,
        strategies=list(strategies),
        n_steps=n_steps,
        step_scale=step_scale,
        lr=lr,
    )
    strategy_summary = summarize_by_strategy(sessions)
    win_rates = win_rates_vs_baseline(sessions, baseline="random")
    sessions_with_meta = merge_sessions_with_user_metadata(sessions, dataset)
    return sessions, curves, strategy_summary, win_rates, sessions_with_meta


def make_target_mode_summary_table(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    """Compact table for article: one row per target_mode x strategy."""
    cols = [
        "target_mode",
        "strategy",
        "users",
        "mean_final_distance",
        "std_final_distance",
        "mean_best_distance",
        "mean_improvement_pct",
    ]
    df = strategy_summary[cols].copy()
    df["target_mode_label"] = df["target_mode"].map(_target_label)
    df["strategy_label"] = df["strategy"].map(_strategy_label)
    return df[
        [
            "target_mode_label",
            "strategy_label",
            "users",
            "mean_final_distance",
            "std_final_distance",
            "mean_best_distance",
            "mean_improvement_pct",
        ]
    ].rename(columns={
        "target_mode_label": "target_mode",
        "strategy_label": "strategy",
    })


def make_winners_table(strategy_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for target_mode, group in strategy_summary.groupby("target_mode"):
        best = group.sort_values("mean_final_distance", ascending=True).iloc[0]
        rows.append({
            "target_mode": _target_label(target_mode),
            "winner": _strategy_label(best["strategy"]),
            "mean_final_distance": float(best["mean_final_distance"]),
            "mean_best_distance": float(best["mean_best_distance"]),
            "mean_improvement_pct": float(best["mean_improvement_pct"]),
        })
    return pd.DataFrame(rows)


def make_archetype_improvement_table(sessions_with_meta: pd.DataFrame) -> pd.DataFrame:
    """Compare semantic_active_v21 against semantic_control_v21 by archetype."""
    df = sessions_with_meta[
        (sessions_with_meta["target_mode"] == "archetype8d")
        & (sessions_with_meta["strategy"].isin(["semantic_control_v21", "semantic_active_v21"]))
    ].copy()
    summary = (
        df.groupby(["main_archetype", "strategy"], dropna=False)
        .agg(
            users=("user_id", "nunique"),
            mean_final_distance=("final_distance", "mean"),
            mean_best_distance=("best_distance", "mean"),
        )
        .reset_index()
    )
    pivot = summary.pivot(index="main_archetype", columns="strategy", values="mean_final_distance")
    users = summary.groupby("main_archetype")["users"].max()
    pivot["users"] = users
    pivot = pivot.reset_index()
    old = "semantic_control_v21"
    new = "semantic_active_v21"
    pivot["improvement_abs"] = pivot[old] - pivot[new]
    pivot["improvement_pct"] = 100.0 * pivot["improvement_abs"] / (pivot[old] + 1e-8)
    pivot["winner"] = np.where(pivot["improvement_abs"] >= 0, "Semantic active v3", "Semantic 6D v2.1")
    pivot = pivot.sort_values("improvement_pct", ascending=False)
    return pivot.rename(columns={
        "main_archetype": "archetype",
        old: "semantic_6d_v21_final_distance",
        new: "semantic_active_v3_final_distance",
    })


def save_article_tables(
    strategy_summary: pd.DataFrame,
    win_rates: pd.DataFrame,
    sessions_with_meta: pd.DataFrame,
    output_dir: str | Path,
) -> dict[str, Path]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    tables = {
        "strategy_descriptions": STRATEGY_DESCRIPTIONS,
        "target_mode_summary": make_target_mode_summary_table(strategy_summary),
        "target_mode_winners": make_winners_table(strategy_summary),
        "win_rates_vs_random": win_rates.copy(),
        "v3_vs_v21_by_archetype": make_archetype_improvement_table(sessions_with_meta),
    }
    paths: dict[str, Path] = {}
    for name, table in tables.items():
        path = output_dir / f"article_{name}.csv"
        table.to_csv(path, index=False)
        paths[name] = path
    return paths


def plot_mean_final_distance_by_target_mode(
    strategy_summary: pd.DataFrame,
    strategies: list[str] | None = None,
    save_path: str | Path | None = None,
):
    use_article_style()
    strategies = ARTICLE_STRATEGIES if strategies is None else strategies
    df = strategy_summary[strategy_summary["strategy"].isin(strategies)].copy()
    target_modes = _ordered_existing(df["target_mode"].unique(), TARGET_MODE_ORDER)
    table = df.pivot(index="target_mode", columns="strategy", values="mean_final_distance").loc[target_modes]
    table = table[[s for s in strategies if s in table.columns]]
    table.index = [_target_label(x) for x in table.index]
    table.columns = [_strategy_label(x) for x in table.columns]

    fig, ax = plt.subplots(figsize=(12, 5.5), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Среднее финальное расстояние до target по режимам генерации", pad=12, fontweight="bold")
    ax.set_xlabel("Режим генерации synthetic users")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75", ncol=2)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax, table


def plot_mean_convergence(
    curves: dict[str, dict[str, np.ndarray]],
    target_mode: str = "archetype8d",
    strategies: list[str] | None = None,
    save_path: str | Path | None = None,
):
    use_article_style()
    strategies = ["random", "uncertainty_axis", "semantic_control_v21", "semantic_active_v21", "candidate_pool_active", "adaptive_router_v32"] if strategies is None else strategies
    by_strategy = curves[target_mode]
    fig, ax = plt.subplots(figsize=(10.5, 5.5), facecolor="white")
    for strategy in strategies:
        if strategy not in by_strategy:
            continue
        arr = np.asarray(by_strategy[strategy], dtype=float)
        mean = arr.mean(axis=0)
        steps = np.arange(len(mean))
        ax.plot(steps, mean, marker="o", linewidth=2.0, markersize=4, label=_strategy_label(strategy))
    ax.set_title(f"Средняя сходимость A/B-сессии: {_target_label(target_mode)}", pad=12, fontweight="bold")
    ax.set_xlabel("Шаг A/B-сессии")
    ax.set_ylabel("Среднее расстояние до скрытого target")
    ax.grid(True, alpha=0.3, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75", ncol=2)
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax


def plot_win_rates_vs_random(
    win_rates: pd.DataFrame,
    target_mode: str = "archetype8d",
    strategies: list[str] | None = None,
    metric: str = "win_rate_final_distance",
    save_path: str | Path | None = None,
):
    use_article_style()
    strategies = ["uncertainty_axis", "semantic_control_v21", "semantic_active_v21", "candidate_pool_active", "adaptive_router_v32"] if strategies is None else strategies
    df = win_rates[(win_rates["target_mode"] == target_mode) & (win_rates["strategy"].isin(strategies))].copy()
    df["label"] = df["strategy"].map(_strategy_label)
    df = df.set_index("label").loc[[_strategy_label(s) for s in strategies if _strategy_label(s) in df["label"].values]].reset_index()
    values = df[metric].values * 100.0

    fig, ax = plt.subplots(figsize=(10, 5), facecolor="white")
    ax.bar(df["label"], values)
    ax.axhline(50.0, linestyle="--", linewidth=1.2)
    ax.set_title(f"Win-rate относительно Random direction: {_target_label(target_mode)}", pad=12, fontweight="bold")
    ax.set_xlabel("Pair Generator")
    ax.set_ylabel("Win-rate, %")
    ax.set_ylim(0, max(100, float(values.max()) + 10))
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    for idx, value in enumerate(values):
        ax.text(idx, value + 1.5, f"{value:.0f}%", ha="center", va="bottom", fontsize=10)
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax, df


def plot_final_distance_boxplot(
    sessions: pd.DataFrame,
    target_mode: str = "archetype8d",
    strategies: list[str] | None = None,
    save_path: str | Path | None = None,
):
    use_article_style()
    strategies = ["random", "uncertainty_axis", "semantic_control_v21", "semantic_active_v21", "candidate_pool_active", "adaptive_router_v32"] if strategies is None else strategies
    df = sessions[(sessions["target_mode"] == target_mode) & (sessions["strategy"].isin(strategies))].copy()
    data = [df[df["strategy"] == s]["final_distance"].values for s in strategies]
    labels = [_strategy_label(s) for s in strategies]
    fig, ax = plt.subplots(figsize=(11, 5.5), facecolor="white")
    ax.boxplot(data, labels=labels, showmeans=True)
    ax.set_title(f"Распределение final distance: {_target_label(target_mode)}", pad=12, fontweight="bold")
    ax.set_xlabel("Pair Generator")
    ax.set_ylabel("Final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.tick_params(axis="x", rotation=25)
    for label in ax.get_xticklabels():
        label.set_ha("right")
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax


def plot_intensity_analysis(
    sessions_with_meta: pd.DataFrame,
    strategies: list[str] | None = None,
    save_path: str | Path | None = None,
):
    use_article_style()
    strategies = ["random", "semantic_control_v21", "semantic_active_v21", "candidate_pool_active", "adaptive_router_v32"] if strategies is None else strategies
    df = sessions_with_meta[
        (sessions_with_meta["target_mode"] == "archetype8d")
        & (sessions_with_meta["strategy"].isin(strategies))
    ].copy()
    group = summarize_by_group(df, ["intensity_label"])
    table = group.pivot(index="intensity_label", columns="strategy", values="mean_final_distance")
    order = [x for x in INTENSITY_ORDER if x in table.index]
    table = table.loc[order]
    table = table[[s for s in strategies if s in table.columns]]
    table.columns = [_strategy_label(s) for s in table.columns]
    fig, ax = plt.subplots(figsize=(11, 5.2), facecolor="white")
    table.plot(kind="bar", ax=ax, width=0.82)
    ax.set_title("Mean final distance по intensity-группам: Archetype 8D", pad=12, fontweight="bold")
    ax.set_xlabel("Intensity group")
    ax.set_ylabel("Mean final distance в weighted 8D")
    ax.grid(True, axis="y", alpha=0.3, linestyle="--")
    ax.legend(title="Pair Generator", frameon=True, facecolor="white", edgecolor="0.75", ncol=2)
    ax.tick_params(axis="x", rotation=0)
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax, table


def plot_archetype_v3_improvement(
    improvement_table: pd.DataFrame,
    save_path: str | Path | None = None,
):
    use_article_style()
    table = improvement_table.copy().sort_values("improvement_pct", ascending=True)
    labels = table["archetype"].astype(str).values
    values = table["improvement_pct"].values
    fig, ax = plt.subplots(figsize=(10, 6.5), facecolor="white")
    ax.barh(labels, values)
    ax.axvline(0.0, linestyle="--", linewidth=1.2)
    ax.set_title("Улучшение Semantic active v3 относительно Semantic 6D v2.1", pad=12, fontweight="bold")
    ax.set_xlabel("Улучшение mean final distance, %")
    ax.set_ylabel("Main archetype")
    ax.grid(True, axis="x", alpha=0.3, linestyle="--")
    for idx, value in enumerate(values):
        offset = 0.8 if value >= 0 else -0.8
        ha = "left" if value >= 0 else "right"
        ax.text(value + offset, idx, f"{value:.1f}%", va="center", ha=ha, fontsize=9)
    plt.tight_layout()
    if save_path is not None:
        save_figure(fig, str(save_path))
    return fig, ax
