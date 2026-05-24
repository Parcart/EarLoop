# Implementation notes

## Что включено

- `vendor/earloop` — backend/audio исходники из текущего приложения.
- `vendor/personalization` — research-контур персонализации.
- `earloop_cli` — новый CLI layer: меню, аудиомост, сессия, логи, профили, offline eval.

## Default strategy

По умолчанию выбрана стратегия:

```text
phase_aware_feedback
```

Это `phase_mixed_contract_v6` + ручной Directional Feedback + phase-aware обработка feedback/soft-stop.

## Аудиомост

В MVP используется прямой мост:

```text
capture device -> EqualizerProcessor -> playback device
```

Для системного звука обычно нужен VB-Cable:

```text
Windows/system output -> CABLE Input
CLI capture device -> CABLE Output
CLI playback device -> headphones/speakers
```

## Mapper

В пакет включён `InterpretableContractMapper8D` как runtime-safe mapper. Если нужен именно обученный Model B contract controllable MLP, лучше экспортировать его в TorchScript и подключить через уже имеющийся в `vendor/personalization/contract_mapper.py` адаптер `TorchScriptContractMapper`.

## Что доработать перед отправкой тестерам

1. Проверить audio device selection на Windows.
2. Экспортировать/подключить финальный learned mapper, если он есть.
3. Сделать 2-3 сухих теста `--no-audio`.
4. Сделать 1 локальный тест с VB-Cable.
5. Уточнить текст README.txt для тестеров.
