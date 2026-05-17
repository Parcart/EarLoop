# EQ personalization research

Проектная заготовка для прокачки контура персонализации эквалайзера.

## Идея

На этом этапе реализуется offline simulator:

```text
hidden user target -> A/B pair -> synthetic user choice -> PreferenceState update -> next pair
```

Mapper пока не подключен. Сначала проверяется, сходится ли personalization loop в compact weighted 8D-пространстве.

## Структура

```text
research/
├── personalization/
│   ├── state.py              # PreferenceState, FEATURE_NAMES_8D
│   ├── control_basis.py      # semantic 4D controls mapped into 8D
│   ├── synthetic_user.py     # SyntheticUser with hidden target
│   ├── pair_generator.py     # A/B pair generation strategies
│   ├── preference_update.py  # update by A/B choice + uncertainty update
│   ├── feedback.py           # Directional Feedback shift
│   ├── metrics.py            # distance and summaries
│   ├── plotting.py           # article-ready plots
│   └── loop.py               # full offline loop
├── experiments/
│   ├── 01_personalization_loop_v0.py
│   └── 02_pair_generator_batch_test.py
├── outputs/
│   ├── figures/
│   └── metrics/
├── main.py
└── requirements.txt
```

## Запуск

Из папки `research`:

```bash
pip install -r requirements.txt
python main.py
python experiments/01_personalization_loop_v0.py
python experiments/02_pair_generator_batch_test.py
```

Batch test с параметрами:

```bash
python experiments/02_pair_generator_batch_test.py --n-users 100 --n-steps 25 --noise-std 0.05 --target-mode random8d
python experiments/02_pair_generator_batch_test.py --n-users 100 --n-steps 25 --noise-std 0.05 --target-mode semantic4d
```

## Режимы synthetic target

- `random8d`: скрытый target генерируется произвольно во всем 8D-пространстве. Это проверяет общий случай.
- `semantic4d`: скрытый target генерируется из 4D semantic controls и затем отображается в 8D. Это проверяет гипотезу, что пользовательские предпочтения могут лежать в сжатом семантическом подпространстве.

## Стратегии Pair Generator

- `random`: случайное многомерное направление в 8D.
- `uncertainty_axis`: один 8D-признак, выбранный по неопределенности `z_std`.
- `semantic_control`: одно семантическое 4D-направление, отображаемое в 8D.
- `hybrid`: смесь random, semantic_control и uncertainty_axis.

## Почему semantic controls

Наивная осевая стратегия проверяет только один признак за шаг. Это интерпретируемо, но может быть медленно для многомерного target. Semantic controls сохраняют 8D как внутреннее пространство, но двигают его по более крупным и музыкально осмысленным направлениям: `low_power`, `warmth_body`, `presence_clarity`, `air_brightness`.

## Dataset-based synthetic user experiments

The project now separates synthetic user generation from Pair Generator testing.

### 1. Generate fixed synthetic users dataset

```bash
python experiments/00_generate_synthetic_users_dataset.py --n-per-mode 100 --name synthetic_users_v1
```

This creates:

```text
outputs/datasets/synthetic_users_v1.csv
outputs/datasets/synthetic_users_v1_metadata.json
```

The dataset contains three target modes:

- `random8d` — stress-test targets in full weighted 8D space;
- `semantic4d` — targets generated in 4D semantic control space and mapped to 8D;
- `archetype8d` — more realistic users generated as mixtures of audio archetypes.

Each row stores:

- hidden target vector `z_*`;
- per-feature importance `importance_*`;
- choice noise level `noise_std`;
- target mode metadata.

### 2. Run Pair Generator batch comparison on the fixed dataset

```bash
python experiments/02_pair_generator_batch_test_from_dataset.py --dataset outputs/datasets/synthetic_users_v1.csv --n-steps 25 --prefix v2_dataset_batch
```

This compares:

- `random`
- `uncertainty_axis`
- `semantic_control`
- `hybrid`

and saves result tables to `outputs/metrics/` and article-ready figures to `outputs/figures/`.

### 3. Notebooks

- `notebooks/00_generate_synthetic_users_dataset.ipynb` — interactive dataset generation and sanity checks;
- `notebooks/02_pair_generator_v2_semantic_compare.ipynb` — dataset-based strategy comparison.

Reusable logic has been moved into:

- `personalization/synthetic_dataset.py`
- `personalization/batch_eval.py`

## v2.1 semantic basis comparison

В версии v2.1 добавлена расширенная semantic basis 6D для Pair Generator:

- `low_power` — сила низа;
- `warmth_body` — тепло и тело;
- `presence_clarity` — присутствие и ясность;
- `air_brightness` — воздух и яркость;
- `club_energy` — клубная энергия / бас с умеренной верхней энергией;
- `clean_bass` — чистый бас без избыточной мутности.

Новые стратегии:

- `semantic_control_v21` — semantic control на основе 6D basis;
- `hybrid_v21` — гибрид random + semantic 6D + uncertainty-axis.

Новый ноутбук:

```text
notebooks/03_pair_generator_v21_semantic_compare.ipynb
```

Он сравнивает:

```text
random
uncertainty_axis
semantic_control          # v2 4D
semantic_control_v21      # v2.1 6D
hybrid                    # v2 4D hybrid
hybrid_v21                # v2.1 6D hybrid
```

Новый скрипт для запуска из папки `research`:

```bash
python experiments/03_pair_generator_v21_semantic_compare.py --n-per-mode 100 --n-steps 25 --prefix v21_semantic_compare
```

Графики в ноутбуке оформлены в светлой теме с русскими подписями для использования в статье / дипломе.


## V3 semantic active Pair Generator

V3 adds `semantic_active_v21`, an active semantic question selector. It uses the v2.1 6D semantic basis but does not choose a semantic direction randomly. Instead, it generates several candidate A/B questions and selects the one with the highest question-usefulness score:

- uncertainty coverage over `z_std`;
- A/B diversity;
- safety penalty;
- repetition penalty for repeated semantic controls.

This score is not a Preference Model and does not predict which candidate the user will prefer. It only selects the next most useful A/B question.

Run:

```bash
python experiments/04_pair_generator_v3_semantic_active.py
```

Or open:

```text
notebooks/04_pair_generator_v3_semantic_active.ipynb
```

## V3.1 Candidate Pool Active Pair Generator

Новая стратегия `candidate_pool_active` расширяет V3 `semantic_active_v21`: вместо выбора вопроса только среди semantic-направлений она строит смешанный пул A/B-вопросов из random, uncertainty-axis и semantic v2.1 candidates, затем выбирает вопрос с максимальным `question-usefulness score`.

Запуск эксперимента:

```bash
python experiments/05_pair_generator_candidate_pool_active.py
```

Новый ноутбук:

```text
notebooks/05_pair_generator_candidate_pool_active.ipynb
```

Важно: `candidate_pool_active` всё ещё не является Preference Model. Она не прогнозирует, что выберет пользователь, а выбирает потенциально полезный вопрос по coverage неопределённости, diversity, safety и repetition penalty.

## V3.2 adaptive router

Добавлена стратегия `adaptive_router_v32` для Pair Generator. Это не Preference Model, а лёгкий эвристический router, который на каждом шаге выбирает, какой тип A/B-вопроса задать:

- `semantic_active_v21`, если текущий `z_mean` или недавнее движение хорошо ложится в semantic 6D basis;
- `uncertainty_axis`, если неопределённость `z_std` сконцентрирована на одной raw 8D-оси;
- `candidate_pool_active` как mixed-exploration fallback и warmup.

Запуск эксперимента:

```bash
python experiments/06_pair_generator_adaptive_router.py
```

Ноутбук:

```text
notebooks/06_pair_generator_adaptive_router.ipynb
```

Основная проверка: сравнить `adaptive_router_v32` с `semantic_active_v21` и `candidate_pool_active` на `random8d`, `semantic4d`, `semantic6d` и `archetype8d`.

## V4a: Logistic Preference Model

V4a checks whether a lightweight online Preference Model can learn from A/B choices while Pair Generator remains fixed.

Run from the `research` directory:

```bash
python experiments/08_preference_model_logistic.py
```

Notebook:

```text
notebooks/08_preference_model_logistic.ipynb
```

The experiment compares:

- `Heuristic update`: the existing rule-based update of `z_mean`;
- `Logistic Preference Model`: an online model that learns `z_pref` from pairwise A/B observations.

The fixed Pair Generator is `semantic_active_v21`, so this experiment isolates Preference Model learning before using the model for pair generation.
