from pathlib import Path
import sys

THIS = Path(__file__).resolve()
MAPPER_DIR = THIS.parents[1]
RESEARCH_DIR = MAPPER_DIR.parent
sys.path.insert(0, str(RESEARCH_DIR))

import pandas as pd

from mapper_v2 import mapper_io as mio
from mapper_v2 import mapper_metrics as mm
from mapper_v2 import mapper_models as mmod
from mapper_v2 import mapper_basis as mb


def main():
    output_dir = MAPPER_DIR / "outputs"
    dirs = mio.ensure_dirs(MAPPER_DIR)
    df, freqs = mio.load_available_mapper_sources(output_dir)
    df, normalizer = mio.prepare_scale_aligned_dataset(df, freqs)
    normalizer.save_json(dirs["outputs"] / "mapper_v4_feature_scale_normalizer.json")
    X, Y = mio.dataframe_to_training_arrays(df)
    X_train, Y_train, X_val, Y_val, X_test, Y_test = mmod.split_train_val_test(X, Y, seed=42)
    model, expected_basis = mmod.make_mapper_model("mlp", X.shape[1], Y.shape[1], freqs)
    cfg = mmod.MapperTrainConfig(num_epochs=5, verbose_every=1, seed=42)
    exp = mmod.MapperExperiment(model, X_train, Y_train, X_val, Y_val, X_test, Y_test, freqs, expected_basis, config=cfg)
    exp.fit()
    row = mm.summarize_mapper_quality("smoke_mlp", exp.predict, X_test, Y_test, freqs, normalizer)
    pd.DataFrame([row]).to_csv(dirs["metrics"] / "mapper_v4_smoke_summary.csv", index=False)
    print(row)


if __name__ == "__main__":
    main()
