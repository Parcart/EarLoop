# Метрики, которые CLI собирает для статьи

## Реальные пользователи

Для реальных людей нет hidden `z_target`, поэтому нельзя честно считать distance-to-target. Вместо этого CLI собирает поведенческие и сессионные метрики:

- `completion_rate` — доля завершённых сессий;
- `profile_save_rate` — доля сессий, где пользователь сохранил профиль;
- `steps_completed` — число пройденных A/B-итераций;
- `soft_stop_triggered` — сработал ли stop marker;
- `soft_stop_step` — на каком шаге система предложила завершиться;
- `soft_stop_continue_rate` — как часто пользователь продолжил после рекомендации;
- `feedback_count` — число Directional Feedback событий;
- `feedback_rate` — доля шагов с Directional Feedback;
- `response_time_sec` — время реакции пользователя на каждом шаге;
- `final_rating` — субъективная оценка итогового профиля 1..5;
- `safety_shrink`, `safety_ok`, `pair_max_abs_db` — метрики безопасности A/B-пар;
- `mapped_max_abs_db`, `mapped_rms_db`, `mapped_smoothness` — характеристики итоговых EQ-кривых.

## Динамика по шагам

Каждый шаг сохраняет:

- `z_a`, `z_b` — кандидаты A/B в contract-space;
- `eq_a_db`, `eq_b_db` — 23-полосные EQ-кривые кандидатов;
- `z_mean_before`, `z_mean_after` — состояние preference state до/после ответа;
- `z_std_after` — неопределённость по каждой оси;
- `z_model_after` — online Preference Model state;
- `loss_before`, `loss_after`, `p_before`, `pred_before` — диагностика модели;
- `feedback_label`, `feedback_strength`, `feedback_phase` — события Directional Feedback;
- `pair_source`, `contract_mode`, `pair_distance_z`, `pair_distance_db_rms` — тип и сила вопроса.

## Offline simulation

Команда `eval-strategies` дополнительно считает метрики с hidden target:

- `final_distance_z_mean`;
- `final_distance_db_rms_mean`;
- `ready_rate`;
- `ready_step_mean`;
- `feedback_count_mean`;
- `feedback_rate_mean`.

Именно эти offline-метрики удобно использовать как предварительную проверку стратегии, а реальные пользовательские метрики — как экспериментальную часть статьи.
