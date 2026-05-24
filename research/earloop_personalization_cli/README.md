# EarLoop Personalization CLI Test Runner

Отдельное консольное приложение для тестирования финального контура персонализации EQ с реальными пользователями.

Что умеет:

- запускать A/B-сессию в `z_contract` / weighted 8D space;
- применять A/B EQ-профили через аудиомост на базе backend `earloop.audio`;
- поддерживать Directional Feedback;
- ловить soft-stop marker и предлагать пользователю сохранить профиль или продолжить;
- сохранять профили и включать их позднее;
- писать полные логи: состояния векторов, модели, пары, EQ-кривые и события на каждом шаге;
- сравнивать финальные стратегии в offline simulation;
- собираться в `.exe` через PyInstaller.

## Быстрый запуск из исходников

```bat
cd earloop_personalization_cli
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python main.py
```

Если `main.py` запускается без аргументов, автоматически стартует команда `run`. Это сделано специально для двойного клика по exe и для PyCharm без Script parameters.

## Интерактивный запуск

```bat
python main.py run
```

По умолчанию используется оптимальная стратегия:

```text
phase_aware_feedback
```

Можно выбрать другую:

```bat
python main.py run --strategy semantic_contract_v6
python main.py run --strategy phase_mixed_contract_v6
python main.py run --strategy phase_aware_feedback
```

Без реального аудиомоста, только для проверки логики:

```bat
python main.py run --no-audio
```

## PyCharm Run Configuration

Script path:

```text
...\research\earloop_personalization_cli\main.py
```

Working directory:

```text
...\research\earloop_personalization_cli
```

Script parameters для обычного dry-run:

```text
run --no-audio
```

Script parameters для реального аудио:

```text
run
```

Environment variables, если нужно:

```text
PYTHONUTF8=1
EARLOOP_CLI_CONFIG=config.json
EARLOOP_CLI_DATA_DIR=runtime_data
EARLOOP_CLI_STRATEGY=phase_aware_feedback
EARLOOP_CLI_AUDIO_ENABLED=1
EARLOOP_CLI_AUTO_DETECT_DEVICES=1
```

Если автоопределение устройств ошиблось:

```text
EARLOOP_CLI_CAPTURE_DEVICE=12
EARLOOP_CLI_PLAYBACK_DEVICE=7
```

## Mapper

В текущем тестовом пакете по умолчанию используется режим `mapper.mode = auto`.

- если файл `models/contract_mapper_model_b.torchscript.pt` найден, CLI загружает TorchScript mapper;
- если файл не найден, используется безопасный fallback `InterpretableContractMapper8D`.

Это нужно, чтобы CLI можно было запускать даже без экспортированной нейросетевой модели. Для финального тестового билда положи TorchScript-файл Model B contract controllable MLP сюда:

```text
models/contract_mapper_model_b.torchscript.pt
```

И в `config.json` оставь:

```json
"mapper": {
  "mode": "auto",
  "model_path": "models/contract_mapper_model_b.torchscript.pt",
  "device": "cpu",
  "allow_interpretable_fallback": true
}
```

## Просмотр и применение сохранённых профилей

```bat
python main.py profiles
```

## Offline сравнение стратегий

```bat
python main.py eval-strategies --n-users 80 --budget 24
```

На выходе создаются CSV/JSON в `runtime_data/evaluations/...`.

## Сборка exe

```bat
py -3.11 build_cli_package.py --recreate-venv
```

или:

```bat
build_cli_package_isolated.bat
```

Результат будет в:

```text
release\EarLoop-Personalization-CLI.zip
```

## Что отправлять после теста

После завершения сессии CLI создаёт архив вида:

```text
runtime_data\exports\session_<id>_logs.zip
```

Именно его тестер отправляет исследователю.


## Запуск без аргументов

Если запустить `main.py` или `EarLoopPersonalizationCLI.exe` без аргументов, автоматически стартует команда `run`.
Это сделано специально для double-click exe и PyCharm Run Configuration.

## PyCharm: параметры запуска

Минимально можно оставить `Script path = main.py`, а поле `Parameters` пустым.
Тогда запустится обычная тестовая A/B-сессия.

Удобные варианты Parameters:

```text
run --no-audio
strategies
devices
profiles
eval-strategies --n-users 80 --budget 24
```

Удобные Environment variables для PyCharm:

```text
EARLOOP_CLI_AUDIO_ENABLED=0
EARLOOP_CLI_STRATEGY=phase_aware_feedback
EARLOOP_CLI_DATA_DIR=runtime_data
EARLOOP_CLI_CAPTURE_DEVICE=0
EARLOOP_CLI_PLAYBACK_DEVICE=1
EARLOOP_CLI_AUTO_DETECT_DEVICES=1
EARLOOP_CLI_MAPPER_MODE=auto
EARLOOP_CLI_MAPPER_PATH=models/model_b_contract_controllable_mlp.npz
```

Для первого dry-run теста без аудио:

```text
EARLOOP_CLI_AUDIO_ENABLED=0
```

## Mapper runtime

По умолчанию CLI использует обученный mapper:

```text
models/model_b_contract_controllable_mlp.npz
```

Это экспорт Model B contract controllable MLP из PyTorch checkpoint в NumPy weights.
Такой формат нужен, чтобы не тащить PyTorch в tester exe и не раздувать сборку.
Если файл модели не найден, CLI может перейти на безопасный interpretable fallback, но для финальных тестов лучше использовать именно `.npz` модель.
