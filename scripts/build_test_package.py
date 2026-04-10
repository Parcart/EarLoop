from __future__ import annotations

import shutil
import subprocess
import sys
import venv
import os
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
DESKTOP_DIR = UI_DIR / "desktop"
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release"

BACKEND_DIST_DIR = DIST_DIR / "backend"
DESKTOP_DIST_DIR = DIST_DIR / "desktop"
DRIVER_DIST_DIR = DIST_DIR / "driver"
REQUIREMENTS_DIR = ROOT / "requirements"
BACKEND_RUNTIME_REQUIREMENTS = REQUIREMENTS_DIR / "backend-runtime.txt"
BACKEND_BUILD_REQUIREMENTS = REQUIREMENTS_DIR / "backend-build.txt"
BUILD_VENV_DIR = ROOT / ".venv-build"

PYINSTALLER_BUILD_DIR = ROOT / "build" / "pyinstaller"
DESKTOP_BACKEND_STAGING_DIR = DESKTOP_DIR / "backend"

FINAL_DIR = RELEASE_DIR / "EarLoop-TestBuild"
FINAL_ZIP_BASE = RELEASE_DIR / "EarLoop-TestBuild"

RUN_SERVER_PY = ROOT / "run_server.py"
BACKEND_EXE = BACKEND_DIST_DIR / "run_server.exe"
BUILD_PYTHON_ENV_VAR = "EARLOOP_BUILD_PYTHON"

README_TEXT = r"""EarLoop — тестовый билд

Спасибо, что согласились протестировать
Это ранняя версия, сейчас важно проверить стабильность и собрать логи.

Как запустить
1. Распакуйте архив
2. Обязательно установите VB-Cable из папки drivers/ (VBCABLE_Setup_x64.exe)
3. После установки:
    ПКМ по значку звука → Параметры звука
    найдите CABLE Output (устройства ввода)
    выставьте тот же sample rate / битрейт, что у вашего устройства вывода (наушники/колонки)
4. Запустите EarLoop.exe
5. Пройдите короткое обучение
6. Создайте новый профиль и попробуйте добиться комфортного звука
7. Отправить логи

Как отправить логи
1. Нажмите Win + R Вставьте: 

%APPDATA%\EarLoop\engine-runtime

2. Откройте папку logs
3. Отправьте все файлы мне в личные сообщения

Важно:
- без установленного VB-Cable приложение будет работать некорректно
- sample rate/битрейт у CABLE Output и конечного устройства вывода должны совпадать

Это первый тестовый билд, буду очень рад получить дополнительный фидбек: 

По интерфейсу: что не понравилось, проблемы с не правильным расположением элементов, визуальные баги;
По генерации подборок: чего на ваш взгляд не хватает, получилось ли у вас достичь результата;
По работе аудио-движка: если не удалось вывести звук опишите проблему, присутствуют ли постоянные заикания при активном окне.

Качество подбора сейчас на раннем уровне -> ваши логи и фидбек напрямую помогут улучшить модель

"""


def find_running_processes_for_path(path: Path) -> list[tuple[int, str]]:
    if sys.platform != "win32":
        return []

    escaped_path = str(path).replace("\\", "\\\\").replace("'", "''")
    command = (
        "Get-CimInstance Win32_Process "
        f"| Where-Object {{ $_.ExecutablePath -eq '{escaped_path}' }} "
        "| Select-Object ProcessId, Name "
        "| ConvertTo-Json -Compress"
    )
    result = subprocess.run(
        ["powershell", "-NoProfile", "-Command", command],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0 or not result.stdout.strip():
        return []

    try:
        import json

        payload = json.loads(result.stdout)
    except Exception:
        return []

    if isinstance(payload, dict):
        payload = [payload]

    processes: list[tuple[int, str]] = []
    for item in payload:
        try:
            processes.append((int(item["ProcessId"]), str(item["Name"])))
        except Exception:
            continue
    return processes


def ensure_executable_is_not_running(path: Path) -> None:
    processes = find_running_processes_for_path(path)
    if not processes:
        return

    process_list = ", ".join(f"{name} (PID {pid})" for pid, name in processes)
    raise RuntimeError(
        f"Файл занят запущенным процессом: {path}\n"
        f"Сейчас используют: {process_list}\n"
        "Закройте приложение EarLoop и все процессы run_server.exe, затем повторите сборку."
    )

def run(cmd: list[str], cwd: Path | None = None) -> None:
    pretty_cwd = str(cwd or ROOT)
    print(f"\n>>> [{pretty_cwd}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        ensure_executable_is_not_running(path)
        path.unlink()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def resolve_managed_build_python() -> Path:
    return BUILD_VENV_DIR / "Scripts" / "python.exe" if sys.platform == "win32" else BUILD_VENV_DIR / "bin" / "python"


def resolve_build_python() -> Path:
    override_value = os.environ.get(BUILD_PYTHON_ENV_VAR, "").strip()
    if override_value:
        override_python = Path(override_value)
        return override_python
    return resolve_managed_build_python()


def ensure_requirements_files_exist() -> None:
    missing_files = [path for path in (BACKEND_RUNTIME_REQUIREMENTS, BACKEND_BUILD_REQUIREMENTS) if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Не найдены requirements-файлы для backend packaging: {missing}")


def ensure_build_venv(build_python: Path) -> None:
    if build_python.exists():
        return
    print(f"== Creating backend packaging environment at {BUILD_VENV_DIR} ==")
    builder = venv.EnvBuilder(with_pip=True, clear=False, upgrade_deps=False)
    builder.create(BUILD_VENV_DIR)
    if not build_python.exists():
        raise FileNotFoundError(f"Не удалось создать packaging interpreter: {build_python}")


def install_backend_packaging_requirements(build_python: Path) -> None:
    ensure_requirements_files_exist()
    print(f"== Using backend packaging interpreter: {build_python} ==")
    print("== Upgrading packaging pip ==")
    run([str(build_python), "-m", "pip", "install", "--upgrade", "pip"])
    print("== Installing backend runtime requirements ==")
    run([str(build_python), "-m", "pip", "install", "-r", str(BACKEND_RUNTIME_REQUIREMENTS)])
    print("== Installing backend build requirements ==")
    run([str(build_python), "-m", "pip", "install", "-r", str(BACKEND_BUILD_REQUIREMENTS)])


def run_backend_dependency_sanity_check(build_python: Path) -> None:
    print("== Running backend dependency sanity check ==")
    sanity_script = (
        "import importlib\n"
        "required = ['soundcard', 'sounddevice', 'soundfile', 'numpy', 'scipy', 'pandas', 'PyInstaller']\n"
        "missing = []\n"
        "for name in required:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name}: {exc}')\n"
        "if missing:\n"
        "    raise SystemExit('Missing critical backend packaging deps: ' + '; '.join(missing))\n"
        "print('Backend packaging deps OK')\n"
    )
    run([str(build_python), "-c", sanity_script])


def copy_tree_contents(source: Path, destination: Path) -> None:
    if not source.exists():
        raise FileNotFoundError(f"Не найдена папка с драйверами: {source}")

    ensure_dir(destination)
    for item in source.iterdir():
        target = destination / item.name
        if item.is_dir():
            shutil.copytree(item, target, dirs_exist_ok=True)
        else:
            shutil.copy2(item, target)


def find_desktop_exe() -> Path:
    candidates = sorted(DESKTOP_DIST_DIR.glob("*.exe"))
    candidates = [p for p in candidates if p.is_file()]
    if not candidates:
        raise FileNotFoundError(
            f"Не найден desktop exe в {DESKTOP_DIST_DIR}. "
            "Проверь electron-builder output."
        )
    # Берем самый свежий по времени изменения
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def clean() -> None:
    print("== Cleaning old build artifacts ==")
    remove_path(BACKEND_DIST_DIR)
    remove_path(DESKTOP_DIST_DIR)
    remove_path(FINAL_DIR)
    remove_path(FINAL_ZIP_BASE.with_suffix(".zip"))
    remove_path(PYINSTALLER_BUILD_DIR)
    remove_path(DESKTOP_BACKEND_STAGING_DIR / "run_server.exe")


def build_ui() -> None:
    print("== Building UI ==")
    run(["npm.cmd", "run", "build"], cwd=UI_DIR)


def build_backend() -> None:
    print("== Building backend exe ==")
    ensure_dir(BACKEND_DIST_DIR)
    ensure_dir(PYINSTALLER_BUILD_DIR)
    ensure_executable_is_not_running(BACKEND_EXE)
    build_python = resolve_build_python()
    if build_python == resolve_managed_build_python():
        ensure_build_venv(build_python)
    elif not build_python.exists():
        raise FileNotFoundError(
            f"Interpreter from {BUILD_PYTHON_ENV_VAR} not found: {build_python}"
        )
    install_backend_packaging_requirements(build_python)
    run_backend_dependency_sanity_check(build_python)

    print("== Running PyInstaller with backend packaging interpreter ==")
    run(
        [
            str(build_python),
            "-m",
            "PyInstaller",
            str(RUN_SERVER_PY),
            "--onefile",
            "--distpath",
            str(BACKEND_DIST_DIR),
            "--workpath",
            str(PYINSTALLER_BUILD_DIR),
            "--specpath",
            str(PYINSTALLER_BUILD_DIR),
            "--name",
            "run_server",
        ],
        cwd=ROOT,
    )

    if not BACKEND_EXE.exists():
        raise FileNotFoundError(f"Не найден backend exe: {BACKEND_EXE}")


def stage_backend_for_electron() -> None:
    print("== Staging backend for Electron ==")
    ensure_dir(DESKTOP_BACKEND_STAGING_DIR)
    ensure_executable_is_not_running(DESKTOP_BACKEND_STAGING_DIR / "run_server.exe")
    shutil.copy2(BACKEND_EXE, DESKTOP_BACKEND_STAGING_DIR / "run_server.exe")


def build_desktop() -> None:
    print("== Building Electron desktop app ==")
    run(["npm.cmd", "install"], cwd=DESKTOP_DIR)
    run(["npm.cmd", "run", "build"], cwd=DESKTOP_DIR)


def create_final_package() -> Path:
    print("== Creating final tester package ==")
    ensure_dir(RELEASE_DIR)
    ensure_dir(FINAL_DIR)
    ensure_dir(FINAL_DIR / "backend")
    ensure_dir(FINAL_DIR / "data")
    ensure_dir(FINAL_DIR / "logs")
    ensure_dir(FINAL_DIR / "drivers")

    desktop_exe = find_desktop_exe()
    shutil.copy2(desktop_exe, FINAL_DIR / "EarLoop.exe")
    ensure_executable_is_not_running(FINAL_DIR / "backend" / "run_server.exe")
    shutil.copy2(BACKEND_EXE, FINAL_DIR / "backend" / "run_server.exe")
    copy_tree_contents(DRIVER_DIST_DIR, FINAL_DIR / "drivers")

    readme_path = FINAL_DIR / "README.txt"
    readme_path.write_text(README_TEXT, encoding="utf-8")

    archive_path = shutil.make_archive(
        str(FINAL_ZIP_BASE),
        "zip",
        root_dir=FINAL_DIR.parent,
        base_dir=FINAL_DIR.name,
    )
    return Path(archive_path)


def main() -> None:
    print(f"Project root: {ROOT}")
    print(f"Requirements dir: {REQUIREMENTS_DIR}")
    clean()
    build_ui()
    build_backend()
    stage_backend_for_electron()
    build_desktop()
    archive_path = create_final_package()

    print("\n== Done ==")
    print(f"Backend exe: {BACKEND_EXE}")
    print(f"Desktop dist: {DESKTOP_DIST_DIR}")
    print(f"Final folder: {FINAL_DIR}")
    print(f"Final zip:    {archive_path}")


if __name__ == "__main__":
    main()
