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

## V6 contract personalization contour

New files added for the contract-scale contour:

- `contract_space.py` — explicit z-contract scale: z=0 neutral, z=1 noticeable, z=2 extreme.
- `contract_mapper.py` — interpretable 8D -> 23-band mapper and optional learned mapper adapters.
- `contract_pair_generator.py` — wrapper around existing semantic/direct generators with mapped-dB safety checks.
- `contract_session.py` — end-to-end synthetic A/B sessions in z_contract with both z-distance and mapped dB metrics.
- `contract_feedback.py` — rule-based directional feedback updates in z_contract.
- `contract_metrics.py` — mapped curve/pair/session metrics.

The previous notebooks remain historical baselines. Use notebooks `21`, `22`, and `23` for the new contract contour.
