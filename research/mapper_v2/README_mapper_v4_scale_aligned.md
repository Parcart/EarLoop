# Mapper v4 — contract scale aligned mapper

This branch keeps the existing mapper architectures close to the original mapper-v3 experiments, but fixes the input-scale contract.

## Main change

Old v4 draft used percentile scaling:

```text
p5 dataset feature  -> z=-2
median              -> z=0
p95 dataset feature -> z=+2
```

That was useful for reconstruction benchmarks, but wrong for product personalization: a small `-2 dB` bass cut could become `z=-2` if it was near the dataset edge.

The new `mapper_v4_01_scale_aligned_dataset.ipynb` uses **contract dB scaling**:

```text
raw weighted feature dB = +extreme_db[feature] -> z=+2
raw weighted feature dB = -extreme_db[feature] -> z=-2
```

Default contract:

```text
sub_bass   16 dB
bass       14 dB
lowmid      6 dB
warmth      6 dB
presence    5 dB
clarity     7 dB
air        12 dB
brightness  8 dB
```

Example:

```text
raw bass = -2 dB
extreme bass = 14 dB
z_bass = 2 * (-2 / 14) ≈ -0.29
```

## Dataset composition

The v4.1 notebook builds a mixed training dataset:

```text
real SocialFX / OPRA / external curves
+ synthetic contract axis sweeps
+ synthetic archetype presets
+ synthetic safe random profiles
```

Real data teaches reconstruction realism near mild/moderate corrections. Synthetic data teaches the learned mapper what `z≈±2` means in the product contract scale.

## Output files

The notebook saves:

```text
outputs/mapper_v4_scale_aligned_dataset.parquet
outputs/mapper_v4_feature_scale_normalizer.json
outputs/tables/mapper_v4_01_*.csv
outputs/figures/mapper_v4_01_*.png
```

For backward compatibility with the existing training notebook, the contract vector is stored as:

```text
z8_scaled_vector == z8_contract_vector
```

So `mapper_v4_03_train_scale_aligned_mapper.ipynb` can continue using `x_col="z8_scaled_vector"`, but the values now have the correct contract-scale meaning.

## Training configs

`mapper_v4_03_train_scale_aligned_mapper.ipynb` should stay close to the original mapper-v3 configs:

```text
batch_size=64
num_epochs=250
lambda_smooth=0.0
lambda_ctrl=0.003 for controllable/hybrid models
```

The intended comparison is: same architecture/objective style, corrected input scale and broader coverage.
