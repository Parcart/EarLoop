"""I/O helpers for mapper_v2 experiments."""

from __future__ import annotations
from . import mapper_basis as mb

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd

from .feature_space import (
    FEATURE_NAMES_8D,
    FREQS_23_DEFAULT,
    CONTRACT_EXTREME_DB_DEFAULT,
    prepare_curve_columns,
    add_8d_features,
    add_scaled_8d_features,
    FeatureScaleNormalizer,
)
from .mapper_basis import InterpretableMapper8D, make_archetype_presets

ARCHETYPE_Z8_PRESETS = {
    "basshead": {
        "sub_bass": 1.65,
        "bass": 1.45,
        "lowmid": -0.25,
        "warmth": 0.15,
        "presence": -0.20,
        "clarity": 0.10,
        "air": 0.25,
        "brightness": -0.35,
    },
    "warm_dark": {
        "sub_bass": 0.65,
        "bass": 0.80,
        "lowmid": 0.50,
        "warmth": 1.15,
        "presence": -0.55,
        "clarity": -0.65,
        "air": -0.75,
        "brightness": -1.10,
    },
    "bright_air": {
        "sub_bass": -0.35,
        "bass": -0.25,
        "lowmid": -0.35,
        "warmth": -0.20,
        "presence": 0.55,
        "clarity": 1.10,
        "air": 1.55,
        "brightness": 1.35,
    },
    "v_shape": {
        "sub_bass": 1.15,
        "bass": 0.95,
        "lowmid": -0.85,
        "warmth": -0.45,
        "presence": 0.35,
        "clarity": 0.75,
        "air": 1.00,
        "brightness": 0.65,
    },
    "vocal_clear": {
        "sub_bass": -0.25,
        "bass": -0.15,
        "lowmid": -0.30,
        "warmth": 0.10,
        "presence": 1.10,
        "clarity": 0.95,
        "air": 0.45,
        "brightness": 0.35,
    },
    "soft_warm": {
        "sub_bass": 0.30,
        "bass": 0.40,
        "lowmid": 0.35,
        "warmth": 0.85,
        "presence": -0.10,
        "clarity": -0.15,
        "air": 0.05,
        "brightness": -0.20,
    },
    "lowmid_cut": {
        "sub_bass": 0.45,
        "bass": 0.30,
        "lowmid": -1.25,
        "warmth": -0.65,
        "presence": 0.35,
        "clarity": 0.45,
        "air": 0.30,
        "brightness": 0.25,
    },
}


CURVE_COLUMN_CANDIDATES = [
    "curve_23",
    "curve_23_json",
    "curve",
    "curve_json",
    "eq_curve",
    "eq_curve_json",
    "curve_db",
    "curve_db_json",
    "target_curve",
    "target_curve_json",
    "target_eq_curve",
    "target_eq_curve_json",
    "gains_db",
    "gains_db_json",
    "gain_db",
    "gain_db_json",
    "eq_gains",
    "eq_gains_json",
    "profile_curve",
    "profile_curve_json",
    "y",
    "target",
]

FREQ_COLUMN_CANDIDATES = [
    "freqs_23",
    "freqs_23_json",
    "freqs",
    "freqs_json",
    "freq_hz",
    "freq_hz_json",
    "freqs_hz",
    "freqs_hz_json",
    "frequency_hz",
    "frequency_hz_json",
    "frequencies_hz",
    "frequencies_hz_json",
    "bands_hz",
    "bands_hz_json",
]


CURVE_FILE_PRIORITY = [
    "socialfx_curves_with_8d.parquet",
    "socialfx_curves_norm_with_6d.parquet",
    "opra_eq_profiles_with_8d.parquet",
    "opra_eq_profiles.parquet",
    "external_eq_profiles_with_8d.parquet",
    "external_eq_profiles.parquet",
    "combined_8d_features.parquet",
    "socialfx_opra_8d_features.parquet",
    "socialfx_curves_with_6d.parquet",
]

# Files produced by mapper_v4 notebooks should not be fed back as raw real sources.
# Otherwise v4_01 can recursively load a previously mixed real+synthetic dataset
# and rows from different files may contain missing z8_raw_vector values.
GENERATED_DATASET_NAME_PATTERNS = [
    "mapper_v4_scale_aligned_dataset",
    "mapper_v4_contract_dataset",
    "mapper_v4_feature_scale_normalizer",
    "mapper_v4_",
]


EXCLUDED_DATASET_PATTERNS = [
    "external_eq_profiles",
    "mapper_v4_",
    "feature_space",
    "normalizer",
    "combined_8d_features",
    "socialfx_opra_8d_features",

    # avoid duplicate OPRA loads
    "opra_eq_profiles.csv",
    "opra_eq_profiles_with_8d.parquet",

    # avoid duplicate SocialFX loads
    "socialfx_curves_with_6d.parquet",
]

def _is_generated_mapper_artifact(path: str | Path) -> bool:
    name = Path(path).name.lower()
    return any(pattern in name for pattern in GENERATED_DATASET_NAME_PATTERNS)


def find_project_root(start: str | Path | None = None) -> Path:
    path = Path(start or Path.cwd()).resolve()
    for candidate in [path, *path.parents]:
        if (candidate / "research").exists() or (candidate / "outputs").exists() or (candidate / "mapper_v2.ipynb").exists():
            return candidate
    return path


def ensure_dirs(base_dir: str | Path) -> dict[str, Path]:
    base = Path(base_dir)
    dirs = {
        "outputs": base / "outputs",
        "figures": base / "outputs" / "figures",
        "tables": base / "outputs" / "tables",
        "metrics": base / "outputs" / "metrics",
        "models": base / "outputs" / "models",
    }
    for p in dirs.values():
        p.mkdir(parents=True, exist_ok=True)
    return dirs


def _read_dataframe(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path, low_memory=False)
    raise ValueError(f"Unsupported dataset extension: {path.suffix}")


def _standardize_curve_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Map older mapper notebook column names to curve_23/freqs_23 when possible.

    Some files in outputs/tables are feature-only tables. Those should be skipped by
    load_available_mapper_sources rather than crashing the notebook.
    """
    df = df.copy()

    if "curve_23" not in df.columns and "curve_23_json" not in df.columns:
        for col in CURVE_COLUMN_CANDIDATES:
            if col in df.columns:
                if col.endswith("_json"):
                    df["curve_23_json"] = df[col]
                else:
                    df["curve_23"] = df[col]
                break

    if "freqs_23" not in df.columns and "freqs_23_json" not in df.columns:
        for col in FREQ_COLUMN_CANDIDATES:
            if col in df.columns:
                if col.endswith("_json"):
                    df["freqs_23_json"] = df[col]
                else:
                    df["freqs_23"] = df[col]
                break

    return df


def has_curve_column(path: str | Path) -> tuple[bool, list[str]]:
    """Cheaply inspect whether a parquet/csv has a known EQ-curve column."""
    path = Path(path)
    try:
        if path.suffix.lower() in [".parquet", ".pq"]:
            cols = list(pd.read_parquet(path, columns=[]).columns)
            # Some engines return [] for columns=[]; fall back to metadata read.
            if not cols:
                cols = list(pd.read_parquet(path).columns)
        elif path.suffix.lower() == ".csv":
            cols = list(pd.read_csv(path, nrows=0).columns)
        else:
            return False, []
    except Exception:
        try:
            cols = list(_read_dataframe(path).columns)
        except Exception:
            return False, []
    has_curve = any(c in cols for c in CURVE_COLUMN_CANDIDATES)
    return has_curve, cols


def load_curve_dataset(path: str | Path, default_freqs: Iterable[float] | None = None) -> tuple[pd.DataFrame, np.ndarray]:
    df = _read_dataframe(path)
    df = _standardize_curve_column_names(df)
    df, freqs = prepare_curve_columns(df, default_freqs=default_freqs)
    return df, freqs


def _dataset_search_roots(output_dir: str | Path) -> list[Path]:
    root = Path(output_dir)
    roots = [root]
    if (root / "tables").exists():
        roots.append(root / "tables")
    if root.name == "tables" and root.parent.exists():
        roots.append(root.parent)
    # Keep order and remove duplicates.
    seen = set()
    unique = []
    for r in roots:
        rp = r.resolve()
        if rp not in seen:
            unique.append(r)
            seen.add(rp)
    return unique


def _candidate_dataset_paths(output_dir: str | Path) -> list[Path]:
    roots = _dataset_search_roots(output_dir)
    candidates: list[Path] = []

    for root in roots:
        for name in CURVE_FILE_PRIORITY:
            candidates.append(root / name)

    for root in roots:
        if root.exists():
            candidates.extend(sorted(root.glob("*.parquet")))
            candidates.extend(sorted(root.glob("*.csv")))

    candidates = [path for path in candidates if not _is_generated_mapper_artifact(path)]

    # Keep order and remove duplicates.
    seen = set()
    unique = []
    for path in candidates:
        rp = path.resolve()
        if rp not in seen:
            unique.append(path)
            seen.add(rp)
    return unique


def load_available_mapper_sources(
    output_dir: str | Path = "outputs",
    *,
    verbose: bool = True,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load SocialFX/OPRA mapper datasets if present, otherwise create demo data.

    The loader is intentionally tolerant: older mapper notebooks produced both
    curve datasets and feature-only summary tables. Feature-only files are skipped.
    It searches both `output_dir` and `output_dir/tables`.
    """
    frames: list[pd.DataFrame] = []
    freqs: np.ndarray | None = None
    skipped: list[tuple[str, str]] = []

    for path in _candidate_dataset_paths(output_dir):
        name = path.name.lower()

        if any(pattern in name for pattern in EXCLUDED_DATASET_PATTERNS):
            skipped.append((path.name, "excluded by pattern"))
            continue

        if not path.exists() or not path.is_file():
            continue
        try:
            default_freqs = freqs if freqs is not None else FREQS_23_DEFAULT
            df, f = load_curve_dataset(path, default_freqs=default_freqs)
        except ValueError as exc:
            msg = str(exc)
            if "curve_23" in msg or "Unsupported dataset extension" in msg:
                skipped.append((path.name, "no curve_23-compatible column"))
                continue
            raise
        except Exception as exc:
            skipped.append((path.name, f"load failed: {type(exc).__name__}: {exc}"))
            continue

        if len(df) == 0:
            skipped.append((path.name, "empty dataframe"))
            continue
        if "source" not in df.columns:
            df["source"] = path.stem
        df["source_file"] = path.name
        frames.append(df)
        freqs = np.asarray(f, dtype=np.float64)

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        if verbose:
            loaded_files = sorted(set(combined["source_file"].astype(str)))
            print(f"Loaded mapper curve datasets: {loaded_files}")
            if skipped:
                preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:6])
                more = "..." if len(skipped) > 6 else ""
                print(f"Skipped non-curve/invalid files: {preview}{more}")
        return combined, np.asarray(freqs, dtype=np.float64)

    if verbose and skipped:
        preview = ", ".join(f"{name} ({reason})" for name, reason in skipped[:8])
        print(f"No curve datasets loaded. Skipped: {preview}")
        print("Falling back to demo mapper dataset.")
    return make_demo_dataset(n=512)


def make_demo_dataset(n: int = 512, seed: int = 42) -> tuple[pd.DataFrame, np.ndarray]:
    """Create a small synthetic mapper dataset so notebooks run without private data."""
    rng = np.random.default_rng(seed)
    freqs = FREQS_23_DEFAULT.copy()
    mapper = InterpretableMapper8D(freqs_hz=freqs)
    presets = list(make_archetype_presets().values())
    rows = []
    for i in range(n):
        if i < len(presets):
            z = presets[i]
        else:
            z = rng.normal(0, 0.8, size=8).astype(np.float32)
            # Add some extreme coverage.
            if rng.random() < 0.15:
                z[rng.integers(0, 8)] += rng.choice([-1.5, 1.5])
            z = np.clip(z, -2, 2)
        curve = mapper.map_one(z)
        curve = curve + rng.normal(0, 0.15, size=len(freqs)).astype(np.float32)
        rows.append({
            "profile_id": f"demo_{i:04d}",
            "source": "demo_interpretable",
            "curve_23": curve.astype(np.float32),
            "freqs_23": freqs.astype(np.float32),
        })
    return pd.DataFrame(rows), freqs


def prepare_scale_aligned_dataset(
    df: pd.DataFrame,
    freqs: Iterable[float],
    normalizer: FeatureScaleNormalizer | None = None,
    normalizer_lower: float = 5.0,
    normalizer_upper: float = 95.0,
    manual_scale_multipliers: dict[str, float] | None = None,
) -> tuple[pd.DataFrame, FeatureScaleNormalizer]:
    df = df.copy()
    if "z8_raw_vector" not in df.columns:
        df = add_8d_features(df, freqs, curve_col="curve_23", prefix="z8_raw")
    raw = np.stack(df["z8_raw_vector"].values).astype(np.float32)
    if normalizer is None:
        normalizer = FeatureScaleNormalizer.fit_percentile(
            raw,
            lower=normalizer_lower,
            upper=normalizer_upper,
            manual_scale_multipliers=manual_scale_multipliers,
        )
    df = add_scaled_8d_features(df, normalizer, raw_col="z8_raw_vector", prefix="z8_scaled")
    return df, normalizer


def dataframe_to_training_arrays(
    df: pd.DataFrame,
    x_col: str = "z8_scaled_vector",
    y_col: str = "curve_23",
) -> tuple[np.ndarray, np.ndarray]:
    X = np.stack(df[x_col].values).astype(np.float32)
    Y = np.stack(df[y_col].values).astype(np.float32)
    return X, Y



def infer_source_group(row_or_source) -> str:
    """Coarse source labels used for mixed mapper datasets."""
    if isinstance(row_or_source, pd.Series):
        text = " ".join(str(row_or_source.get(c, "")) for c in ["source", "source_file", "dataset", "name"])
    else:
        text = str(row_or_source)
    low = text.lower()
    if "socialfx" in low or "social_fx" in low:
        return "socialfx"
    if "opra" in low:
        return "opra"
    if "external" in low:
        return "external"
    if "synthetic" in low or "demo" in low:
        return "synthetic"
    return "other_real"


def prepare_contract_aligned_dataset(
    df: pd.DataFrame,
    freqs: Iterable[float],
    normalizer: FeatureScaleNormalizer | None = None,
    extreme_db: dict[str, float] | None = None,
    *,
    scaled_prefix: str = "z8_scaled",
) -> tuple[pd.DataFrame, FeatureScaleNormalizer]:
    """Prepare real EQ curves with the explicit dB contract scale.

    This is the intended normalizer for personalization:
        raw weighted feature dB = +/- extreme_db[feature] -> z = +/-2.

    It deliberately does not use dataset percentiles, so a small -2 dB bass cut
    becomes a small negative z rather than an artificial z=-2 edge case.
    """
    df = df.copy()

    # Always recompute raw 8D features from curve_23 for v4 contract datasets.
    # A combined dataframe may contain z8_raw_vector for some source files and NaN
    # for others. Reusing that mixed column causes parse_array_value(NaN) errors
    # and can also keep stale features from old notebooks.
    stale_cols = [
        col for col in df.columns
        if col.startswith("z8_raw") or col.startswith("z8_scaled") or col.startswith("z8_contract")
    ]
    if stale_cols:
        df = df.drop(columns=stale_cols, errors="ignore")
    df = add_8d_features(df, freqs, curve_col="curve_23", prefix="z8_raw")

    if normalizer is None:
        normalizer = FeatureScaleNormalizer.fit_contract(
            extreme_db=extreme_db or CONTRACT_EXTREME_DB_DEFAULT,
            feature_names=FEATURE_NAMES_8D,
            clip_value=2.0,
        )
    df = add_scaled_8d_features(df, normalizer, raw_col="z8_raw_vector", prefix=scaled_prefix)
    # Explicit alias for clarity; downstream notebooks can still use z8_scaled_vector.
    if scaled_prefix == "z8_scaled":
        df["z8_contract_vector"] = df["z8_scaled_vector"]
        for name in FEATURE_NAMES_8D:
            df[f"z8_contract_{name}"] = df[f"z8_scaled_{name}"]
    if "source_group" not in df.columns:
        df["source_group"] = df.apply(infer_source_group, axis=1)
    df["is_synthetic"] = False
    return df, normalizer


def sample_contract_z_random(
    n: int,
    rng: np.random.Generator,
    *,
    feature_names: list[str] | None = None,
    max_abs: float = 2.0,
    strong_probability: float = 0.25,
) -> np.ndarray:
    """Sample safe-ish mixed z profiles in contract scale [-2, 2]."""
    names = feature_names or FEATURE_NAMES_8D
    d = len(names)
    # Mostly moderate values, with occasional strong axes.
    z = rng.normal(0.0, 0.55, size=(n, d)).astype(np.float32)
    strong_mask = rng.random(n) < strong_probability
    for row in np.where(strong_mask)[0]:
        k = int(rng.integers(1, 4))
        idx = rng.choice(d, size=k, replace=False)
        z[row, idx] += rng.choice([-1.0, 1.0], size=k) * rng.uniform(0.7, 1.7, size=k)

    # Avoid pathological all-zones-max cases; contract extremes are better covered by axis/archetype rows.
    z = np.clip(z, -max_abs, max_abs)

    # Small guardrail: if low-end and brightness are both extreme in the same sign,
    # reduce brightness a bit to keep curves listenable.
    if "brightness" in names and "bass" in names:
        ib = names.index("bass")
        ibr = names.index("brightness")
        same_extreme = (np.abs(z[:, ib]) > 1.4) & (np.abs(z[:, ibr]) > 1.4) & (np.sign(z[:, ib]) == np.sign(z[:, ibr]))
        z[same_extreme, ibr] *= 0.65
    return z.astype(np.float32)


def make_axis_contract_z(values: Iterable[float] | None = None, feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES_8D
    vals = list(values if values is not None else [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0])
    rows = []
    for i in range(len(names)):
        for v in vals:
            z = np.zeros(len(names), dtype=np.float32)
            z[i] = float(v)
            rows.append(z)
    return np.stack(rows).astype(np.float32)


def make_synthetic_contract_dataset(
    freqs: Iterable[float],
    *,
    n_random: int = 12000,
    seed: int = 42,
    include_axis: bool = True,
    include_archetypes: bool = True,
    mapper: InterpretableMapper8D | None = None,
    noise_std: float = 0.0,
    source_prefix: str = "synthetic_contract",
) -> pd.DataFrame:
    """Generate synthetic z_contract -> curve_23 pairs to cover strong/extreme regions.

    These rows teach the learned mapper the explicit contract scale. Real SocialFX/OPRA
    rows teach reconstruction realism around mild and moderate corrections.
    """
    rng = np.random.default_rng(seed)
    freqs_arr = np.asarray(freqs, dtype=np.float64)
    mapper = mapper or InterpretableMapper8D(freqs_hz=freqs_arr, safety=True)

    z_parts = []
    source_parts = []

    if include_axis:
        z_axis = make_axis_contract_z(feature_names=mapper.feature_names)
        z_parts.append(z_axis)
        source_parts.extend(["synthetic_axis"] * len(z_axis))

    if include_archetypes:
        presets = make_archetype_presets()
        z_arch = np.stack(list(presets.values())).astype(np.float32)
        z_parts.append(z_arch)
        source_parts.extend(["synthetic_archetype"] * len(z_arch))

    if n_random > 0:
        z_rand = sample_contract_z_random(n_random, rng, feature_names=mapper.feature_names)
        z_parts.append(z_rand)
        source_parts.extend(["synthetic_random"] * len(z_rand))

    z_all = np.concatenate(z_parts, axis=0).astype(np.float32)
    curves = mapper.map_batch(z_all).astype(np.float32)
    if noise_std > 0:
        curves = curves + rng.normal(0.0, float(noise_std), size=curves.shape).astype(np.float32)

    rows = []
    for i, (z, curve, source_group) in enumerate(zip(z_all, curves, source_parts)):
        row = {
            "profile_id": f"{source_prefix}_{i:06d}",
            "source": source_group,
            "source_group": source_group,
            "source_file": "generated_by_mapper_v4_01_contract_dataset",
            "curve_23": curve.astype(np.float32),
            "freqs_23": freqs_arr.astype(np.float32),
            "z8_scaled_vector": z.astype(np.float32),
            "z8_contract_vector": z.astype(np.float32),
            "is_synthetic": True,
        }
        for j, name in enumerate(mapper.feature_names):
            row[f"z8_scaled_{name}"] = float(z[j])
            row[f"z8_contract_{name}"] = float(z[j])
        rows.append(row)
    return pd.DataFrame(rows)


def build_mixed_contract_dataset(
    df_real: pd.DataFrame,
    freqs: Iterable[float],
    normalizer: FeatureScaleNormalizer | None = None,
    extreme_db: dict[str, float] | None = None,
    synthetic_random_n: int | None = None,
    synthetic_fraction_of_real: float = 0.10,
    synthetic_max_random: int = 1200,
    synthetic_archetype_jitter_n: int = 2000,
    synthetic_random_sigma: float = 0.65,
    synthetic_random_extreme_prob: float = 0.20,
    archetype_jitter_sigma: float = 0.22,
    archetype_strength_jitter: float = 0.18,
    seed: int = 42,
    max_real_rows_per_group: dict[str, int] | None = None,
) -> tuple[pd.DataFrame, FeatureScaleNormalizer]:
    """
    Create mixed contract dataset:
    - real OPRA / SocialFX curves with contract-scaled z8;
    - axis sweeps;
    - archetype base profiles;
    - archetype jitter profiles;
    - small safe-random coverage.
    """
    real, normalizer = prepare_contract_aligned_dataset(
        df_real,
        freqs,
        normalizer=normalizer,
        extreme_db=extreme_db,
    )

    rng = np.random.default_rng(seed)

    if max_real_rows_per_group:
        parts = []

        for source_group, g in real.groupby("source_group"):
            max_n = max_real_rows_per_group.get(str(source_group))
            if max_n is not None and len(g) > max_n:
                g = g.sample(n=max_n, random_state=seed)
            parts.append(g)

        real = pd.concat(parts, ignore_index=True)

    real_n = len(real)

    if synthetic_random_n is None:
        synthetic_random_n = int(round(real_n * synthetic_fraction_of_real))
        synthetic_random_n = min(synthetic_random_n, synthetic_max_random)

    synthetic_parts = []

    # 1. Axis sweeps: small but important controllability anchor.
    axis_values = np.asarray(
        [-2.0, -1.5, -1.0, -0.5, 0.5, 1.0, 1.5, 2.0],
        dtype=np.float32,
    )

    axis_z = []

    for feature in FEATURE_NAMES_8D:
        feature_idx = FEATURE_NAMES_8D.index(feature)
        for value in axis_values:
            z = np.zeros(len(FEATURE_NAMES_8D), dtype=np.float32)
            z[feature_idx] = value
            axis_z.append(z)

    axis_z = np.stack(axis_z, axis=0)

    synthetic_parts.append(
        make_synthetic_contract_dataframe(
            axis_z,
            freqs=freqs,
            normalizer=normalizer,
            source_group="synthetic_axis",
            labels=["axis"] * len(axis_z),
            safety=True,
        )
    )

    # 2. Base archetypes.
    archetype_names, archetype_z = build_archetype_z8_matrix()

    synthetic_parts.append(
        make_synthetic_contract_dataframe(
            archetype_z,
            freqs=freqs,
            normalizer=normalizer,
            source_group="synthetic_archetype",
            labels=archetype_names,
            safety=True,
        )
    )

    # 3. Archetype jitter: main synthetic coverage.
    if synthetic_archetype_jitter_n and synthetic_archetype_jitter_n > 0:
        jitter_z, jitter_labels = generate_archetype_jitter_z8_contract(
            n=synthetic_archetype_jitter_n,
            rng=rng,
            jitter_sigma=archetype_jitter_sigma,
            strength_jitter=archetype_strength_jitter,
            clip_value=2.0,
        )

        synthetic_parts.append(
            make_synthetic_contract_dataframe(
                jitter_z,
                freqs=freqs,
                normalizer=normalizer,
                source_group="synthetic_archetype_jitter",
                labels=jitter_labels,
                safety=True,
            )
        )

    # 4. Smaller safe-random coverage.
    if synthetic_random_n and synthetic_random_n > 0:
        random_z = generate_safe_random_z8_contract(
            n=synthetic_random_n,
            rng=rng,
            sigma=synthetic_random_sigma,
            extreme_prob=synthetic_random_extreme_prob,
            max_active_extreme_axes=2,
            clip_value=2.0,
        )

        synthetic_parts.append(
            make_synthetic_contract_dataframe(
                random_z,
                freqs=freqs,
                normalizer=normalizer,
                source_group="synthetic_random",
                labels=["safe_random"] * synthetic_random_n,
                safety=True,
            )
        )

    synthetic = pd.concat(synthetic_parts, ignore_index=True)

    mixed = pd.concat([real, synthetic], ignore_index=True)

    mixed["dataset_kind"] = np.where(
        mixed["is_synthetic"].astype(bool),
        "synthetic",
        "real",
    )

    return mixed, normalizer


def _array_json_value(value) -> str:
    arr = np.asarray(value, dtype=float).reshape(-1)
    return "[" + ",".join(f"{float(v):.10g}" for v in arr) + "]"


def save_curve_dataset_parquet(
    df: pd.DataFrame,
    path: str | Path,
    *,
    array_columns: list[str] | None = None,
    drop_array_columns: bool = True,
) -> Path:
    """Save mapper dataset with JSON array columns for robust cross-notebook loading."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    out = df.copy()
    cols = array_columns or [
        "curve_23",
        "freqs_23",
        "z8_raw_vector",
        "z8_scaled_vector",
        "z8_contract_vector",
    ]
    for col in cols:
        if col in out.columns:
            out[f"{col}_json"] = out[col].apply(_array_json_value)
            if drop_array_columns:
                out = out.drop(columns=[col])
    out.to_parquet(path, index=False)
    return path

def _preset_to_z8_vector(preset: dict[str, float]) -> np.ndarray:
    return np.asarray(
        [preset.get(name, 0.0) for name in FEATURE_NAMES_8D],
        dtype=np.float32,
    )


def build_archetype_z8_matrix() -> tuple[list[str], np.ndarray]:
    names = list(ARCHETYPE_Z8_PRESETS.keys())
    vectors = np.stack(
        [_preset_to_z8_vector(ARCHETYPE_Z8_PRESETS[name]) for name in names],
        axis=0,
    )
    return names, vectors.astype(np.float32)


def generate_safe_random_z8_contract(
    n: int,
    rng: np.random.Generator,
    sigma: float = 0.65,
    extreme_prob: float = 0.20,
    max_active_extreme_axes: int = 2,
    clip_value: float = 2.0,
) -> np.ndarray:
    """
    Safe-ish random profiles:
    - most axes stay near zero;
    - sometimes 1-2 axes get a stronger accent;
    - avoids uniform random over the full 8D cube.
    """
    z = rng.normal(loc=0.0, scale=sigma, size=(n, len(FEATURE_NAMES_8D))).astype(np.float32)

    for row in z:
        if rng.random() < extreme_prob:
            k = int(rng.integers(1, max_active_extreme_axes + 1))
            axes = rng.choice(len(FEATURE_NAMES_8D), size=k, replace=False)
            signs = rng.choice([-1.0, 1.0], size=k)
            accents = rng.uniform(0.75, 1.55, size=k)
            row[axes] += signs * accents

    return np.clip(z, -clip_value, clip_value).astype(np.float32)


def generate_archetype_jitter_z8_contract(
    n: int,
    rng: np.random.Generator,
    jitter_sigma: float = 0.22,
    strength_jitter: float = 0.18,
    clip_value: float = 2.0,
) -> tuple[np.ndarray, list[str]]:
    """
    Generates clouds around meaningful archetype profiles.
    This is safer than pure random because every sample starts from
    a musically interpretable taste direction.
    """
    archetype_names, archetype_z = build_archetype_z8_matrix()

    chosen_idx = rng.integers(0, len(archetype_names), size=n)
    base = archetype_z[chosen_idx].astype(np.float32)

    # Slight global strength variation per profile.
    strength = rng.normal(loc=1.0, scale=strength_jitter, size=(n, 1)).astype(np.float32)
    strength = np.clip(strength, 0.65, 1.35)

    noise = rng.normal(
        loc=0.0,
        scale=jitter_sigma,
        size=base.shape,
    ).astype(np.float32)

    z = base * strength + noise
    z = np.clip(z, -clip_value, clip_value).astype(np.float32)

    labels = [archetype_names[i] for i in chosen_idx]
    return z, labels


def make_synthetic_contract_dataframe(
    z_contract: np.ndarray,
    freqs: Iterable[float],
    normalizer: FeatureScaleNormalizer,
    source_group: str,
    labels: list[str] | None = None,
    safety: bool = True,
) -> pd.DataFrame:
    """
    Converts contract z8 vectors into curve_23 targets using the interpretable mapper.
    Keeps both:
    - z8_scaled_vector / z8_contract_vector: contract input used for training;
    - z8_raw_vector: inverse contract values in dB-like feature space.
    """
    freqs = np.asarray(freqs, dtype=np.float32)
    z_contract = np.asarray(z_contract, dtype=np.float32)

    mapper = mb.InterpretableMapper8D(
        freqs_hz=freqs,
        safety=safety,
    )

    curves = mapper.map_batch(z_contract).astype(np.float32)

    try:
        z_raw = normalizer.inverse_transform(z_contract).astype(np.float32)
    except Exception:
        z_raw = z_contract.astype(np.float32)

    rows = []

    for i in range(len(z_contract)):
        row = {
            "source": source_group,
            "source_group": source_group,
            "synthetic_label": labels[i] if labels is not None else source_group,
            "is_synthetic": True,
            "curve_23": curves[i],
            "freqs_23": freqs,
            "z8_raw_vector": z_raw[i],
            "z8_scaled_vector": z_contract[i],
            "z8_contract_vector": z_contract[i],
        }

        for j, name in enumerate(FEATURE_NAMES_8D):
            row[f"z8_raw_{name}"] = float(z_raw[i, j])
            row[f"z8_scaled_{name}"] = float(z_contract[i, j])
            row[f"z8_contract_{name}"] = float(z_contract[i, j])

        rows.append(row)

    return pd.DataFrame(rows)
