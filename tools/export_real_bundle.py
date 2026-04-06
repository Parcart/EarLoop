from __future__ import annotations

import argparse
import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd


def detect_id_col(df: pd.DataFrame):
    cols = list(df.columns)
    lower_map = {c: str(c).lower() for c in cols}
    for c, lc in lower_map.items():
        if lc == "curve_id":
            return c
    candidates = [
        "id",
        "preset",
        "name",
        "model",
        "device",
        "filename",
        "file",
        "path",
        "brand",
        "headphone",
        "source_file",
    ]
    for c, lc in lower_map.items():
        if lc in candidates:
            return c
    for c, lc in lower_map.items():
        for cand in candidates:
            if cand in lc:
                return c
    return None


def detect_fc_col(df: pd.DataFrame):
    candidates = [
        "fc_hz",
        "freq_hz",
        "frequency_hz",
        "frequency",
        "freq",
        "fc",
        "hz",
    ]
    lower_map = {c: str(c).lower() for c in df.columns}
    for c, lc in lower_map.items():
        if lc in candidates:
            return c
    for c, lc in lower_map.items():
        if ("freq" in lc or "hz" in lc) and "gain" not in lc and "db" not in lc:
            return c
    return None


def detect_gain_col(df: pd.DataFrame):
    candidates = ["gain_db", "gain", "spl", "db", "gain_dbfs", "delta_db"]
    lower_map = {c: str(c).lower() for c in df.columns}
    for c, lc in lower_map.items():
        if lc in candidates:
            return c
    for c, lc in lower_map.items():
        if "gain" in lc or lc.endswith("_db") or lc == "db" or "spl" in lc:
            return c
    return None


def to_long(df: pd.DataFrame, source: str):
    if df is None:
        return None, None
    df = df.copy()
    id_col = detect_id_col(df)
    fc_col = detect_fc_col(df)
    gain_col = detect_gain_col(df)
    if fc_col is None or gain_col is None:
        print(f"[real_bundle] Missing fc/gain columns for {source}. fc={fc_col}, gain={gain_col}")
        return None, id_col
    if id_col is None:
        print(f"[real_bundle] No id column detected for {source}; using index as curve_id.")
        df = df.reset_index().rename(columns={"index": "curve_id"})
        id_col = "curve_id"
    out = df[[id_col, fc_col, gain_col]].rename(
        columns={id_col: "curve_id", fc_col: "fc_hz", gain_col: "gain_db"}
    )
    out["fc_hz"] = pd.to_numeric(out["fc_hz"], errors="coerce")
    out["gain_db"] = pd.to_numeric(out["gain_db"], errors="coerce")
    out = out.dropna(subset=["fc_hz", "gain_db"])
    out["source"] = source
    return out, id_col


def read_table(path: Path) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix in [".csv", ".tsv"]:
        sep = "	" if suffix == ".tsv" else ","
        return pd.read_csv(path, sep=sep)
    if suffix in [".parquet", ".pq"]:
        return pd.read_parquet(path)
    raise ValueError(f"Unsupported table format: {path}")


def load_freqs(path: Path):
    df = read_table(path)
    if df.shape[1] == 1:
        series = df.iloc[:, 0]
    else:
        col = detect_fc_col(df)
        if col is not None:
            series = df[col]
        else:
            numeric = df.select_dtypes(include=["number"])
            if numeric.shape[1] == 0:
                raise ValueError(f"No numeric columns found in {path}")
            series = numeric.iloc[:, 0]
    arr = pd.to_numeric(series, errors="coerce").dropna().to_numpy(dtype=float)
    return arr


def save_freqs(path: Path, freqs):
    arr = np.asarray(freqs, dtype=float)
    arr = arr[~np.isnan(arr)]
    pd.DataFrame({"fc_hz": arr}).to_csv(path, index=False, encoding="utf-8")
    return int(arr.size)


def export_bundle(
    out_dir: Path,
    socialfx_path: Path | None,
    autoeq_path: Path | None,
    final_freqs_path: Path | None,
    grid_freqs_path: Path | None,
):
    out_dir.mkdir(parents=True, exist_ok=True)

    files = []
    manifest = {
        "timestamp": datetime.utcnow().isoformat(timespec="seconds") + "Z",
        "files": files,
    }

    if socialfx_path is not None:
        social_df = read_table(socialfx_path)
        social_long, social_id = to_long(social_df, "socialfx")
        if social_long is not None:
            path = out_dir / "socialfx_long.csv"
            social_long.to_csv(path, index=False, encoding="utf-8")
            files.append(path.name)
            manifest["socialfx"] = {
                "id_col": social_id,
                "n_curves": int(social_long["curve_id"].nunique()),
                "n_rows": int(len(social_long)),
            }
            print(f"[real_bundle] Saved {path} ({len(social_long)} rows)")

    if autoeq_path is not None:
        autoeq_df = read_table(autoeq_path)
        autoeq_long, autoeq_id = to_long(autoeq_df, "autoeq")
        if autoeq_long is not None:
            path = out_dir / "autoeq_long.csv"
            autoeq_long.to_csv(path, index=False, encoding="utf-8")
            files.append(path.name)
            manifest["autoeq"] = {
                "id_col": autoeq_id,
                "n_curves": int(autoeq_long["curve_id"].nunique()),
                "n_rows": int(len(autoeq_long)),
            }
            print(f"[real_bundle] Saved {path} ({len(autoeq_long)} rows)")

    if final_freqs_path is not None:
        freqs_final = load_freqs(final_freqs_path)
        path = out_dir / "freqs_final_23.csv"
        n = save_freqs(path, freqs_final)
        files.append(path.name)
        manifest["freqs_final"] = {"n_freqs": n, "path": path.name}
        print(f"[real_bundle] Saved {path} ({n} freqs)")

    if grid_freqs_path is not None:
        freqs_grid = load_freqs(grid_freqs_path)
        path = out_dir / "freqs_40.csv"
        n = save_freqs(path, freqs_grid)
        files.append(path.name)
        manifest["freqs_40"] = {"n_freqs": n, "path": path.name}
        print(f"[real_bundle] Saved {path} ({n} freqs)")

    manifest_path = out_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, ensure_ascii=True), encoding="utf-8")
    print(f"[real_bundle] Saved {manifest_path}")


def main():
    parser = argparse.ArgumentParser(description="Export real_bundle CSVs and manifest.")
    parser.add_argument("--socialfx", type=Path, help="Path to SocialFX long CSV/Parquet")
    parser.add_argument("--autoeq", type=Path, help="Path to AutoEQ long CSV/Parquet")
    parser.add_argument("--final-freqs", type=Path, help="Path to final freqs CSV/Parquet")
    parser.add_argument("--grid-freqs", type=Path, help="Path to 40-band grid CSV/Parquet")
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("data") / "real_bundle",
        help="Output directory",
    )
    args = parser.parse_args()

    export_bundle(
        args.out_dir,
        args.socialfx,
        args.autoeq,
        args.final_freqs,
        args.grid_freqs,
    )


if __name__ == "__main__":
    main()
