## Сборка desktop-приложения

Сборка Windows-приложения выполняется скриптом:

```text
scripts/build_test_package.py
```

### Требования для сборки

- Windows;
- Node.js и npm;
- Python 3.11;
- Python 3.14;
- Python Launcher `py`, доступный из терминала.

Скрипт использует отдельные версии Python для ML/domain worker и audio bridge worker, создаёт изолированные окружения, собирает интерфейс, backend-компоненты и Electron-приложение.

### Подготовка

Сначала установите зависимости основного интерфейса:

```powershell
cd ui
npm install
cd ..
```

### Запуск сборки

```powershell
python scripts\build_test_package.py
```

После успешной сборки будут созданы:

```text
release/
├── EarLoop-TestBuild/
│   ├── EarLoop.exe
│   ├── data/
│   ├── drivers/
│   ├── logs/
│   └── README.txt
└── EarLoop-TestBuild.zip
```

Скрипт автоматически:

- очищает старые артефакты сборки;
- собирает React/Vite-интерфейс;
- создаёт отдельные Python-окружения для worker-процессов;
- устанавливает runtime- и build-зависимости;
- собирает Python worker-процессы через PyInstaller;
- добавляет backend-компоненты в Electron-приложение;
- собирает desktop-приложение;
- создаёт готовую папку и ZIP-архив в каталоге `release`.

При необходимости пути к интерпретаторам можно задать через переменные окружения:

```powershell
$env:EARLOOP_ML_BUILD_PYTHON = "C:\Path\To\Python311\python.exe"
$env:EARLOOP_AUDIO_BUILD_PYTHON = "C:\Path\To\Python314\python.exe"

python scripts\build_test_package.py
```