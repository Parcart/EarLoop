from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
UI_DIR = ROOT / "ui"
DESKTOP_DIR = UI_DIR / "desktop"
DIST_DIR = ROOT / "dist"
RELEASE_DIR = ROOT / "release"
REQUIREMENTS_DIR = ROOT / "requirements"

BACKEND_DIST_DIR = DIST_DIR / "backend"
DESKTOP_DIST_DIR = DIST_DIR / "desktop"
DRIVER_DIST_DIR = DIST_DIR / "driver"
PYINSTALLER_BUILD_DIR = ROOT / "build" / "pyinstaller"
DESKTOP_BACKEND_STAGING_DIR = DESKTOP_DIR / "backend"

FINAL_DIR = RELEASE_DIR / "EarLoop-TestBuild"
FINAL_ZIP_BASE = RELEASE_DIR / "EarLoop-TestBuild"

LEGACY_BUILD_VENV_DIR = ROOT / ".venv-build"


@dataclass(frozen=True, slots=True)
class BuildTarget:
    key: str
    label: str
    dist_name: str
    entrypoint: Path
    runtime_requirements: Path
    build_requirements: Path
    build_venv_dir: Path
    preferred_python_version: str
    build_python_env_var: str
    sanity_modules: tuple[str, ...]

    @property
    def bundle_dir(self) -> Path:
        return BACKEND_DIST_DIR / self.dist_name

    @property
    def exe_path(self) -> Path:
        return self.bundle_dir / f"{self.dist_name}.exe"

    @property
    def desktop_stage_dir(self) -> Path:
        return DESKTOP_BACKEND_STAGING_DIR / self.dist_name

    @property
    def desktop_stage_exe(self) -> Path:
        return self.desktop_stage_dir / f"{self.dist_name}.exe"

    @property
    def pyinstaller_work_dir(self) -> Path:
        return PYINSTALLER_BUILD_DIR / self.dist_name


ML_TARGET = BuildTarget(
    key="ml_domain_worker",
    label="ML/Domain Worker",
    dist_name="ml_domain_worker",
    entrypoint=ROOT / "run_ml_domain_worker.py",
    runtime_requirements=REQUIREMENTS_DIR / "ml-runtime.txt",
    build_requirements=REQUIREMENTS_DIR / "ml-build.txt",
    build_venv_dir=ROOT / ".venv-build-ml311",
    preferred_python_version="3.11",
    build_python_env_var="EARLOOP_ML_BUILD_PYTHON",
    sanity_modules=("numpy", "colorama", "PyInstaller"),
)

AUDIO_TARGET = BuildTarget(
    key="audio_bridge_worker",
    label="Audio Bridge Worker",
    dist_name="audio_bridge_worker",
    entrypoint=ROOT / "run_audio_bridge_worker.py",
    runtime_requirements=REQUIREMENTS_DIR / "audio-runtime.txt",
    build_requirements=REQUIREMENTS_DIR / "audio-build.txt",
    build_venv_dir=ROOT / ".venv-build-audio314",
    preferred_python_version="3.14",
    build_python_env_var="EARLOOP_AUDIO_BUILD_PYTHON",
    sanity_modules=("numpy", "scipy", "sounddevice", "soundfile", "soundcard", "colorama", "PyInstaller"),
)

BUILD_TARGETS = (ML_TARGET, AUDIO_TARGET)

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


def run(cmd: list[str], cwd: Path | None = None) -> None:
    pretty_cwd = str(cwd or ROOT)
    print(f"\n>>> [{pretty_cwd}] {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd or ROOT)
    if result.returncode != 0:
        raise SystemExit(result.returncode)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


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
        "Закройте приложение EarLoop и worker-процессы, затем повторите сборку."
    )


def remove_path(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        ensure_executable_is_not_running(path)
        path.unlink()


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
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    return candidates[0]


def read_python_version(python_executable: Path) -> str:
    result = subprocess.run(
        [str(python_executable), "-c", "import sys; print(sys.version.split()[0])"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Не удалось определить версию Python для {python_executable}: {result.stderr.strip()}"
        )
    return result.stdout.strip()


def is_expected_python_version(python_executable: Path, expected_prefix: str) -> bool:
    try:
        version = read_python_version(python_executable)
    except Exception:
        return False
    return version.startswith(expected_prefix)


def resolve_managed_build_python(target: BuildTarget) -> Path:
    scripts_dir = "Scripts" if sys.platform == "win32" else "bin"
    return target.build_venv_dir / scripts_dir / ("python.exe" if sys.platform == "win32" else "python")


def resolve_system_python_for_version(version: str) -> Path:
    if sys.platform == "win32":
        launcher = shutil.which("py")
        if launcher:
            result = subprocess.run(
                [launcher, f"-{version}", "-c", "import sys; print(sys.executable)"],
                capture_output=True,
                text=True,
                encoding="utf-8",
                errors="ignore",
            )
            if result.returncode == 0 and result.stdout.strip():
                return Path(result.stdout.strip())
        raise FileNotFoundError(
            f"Не удалось найти системный Python {version} через launcher 'py'."
        )

    for candidate in (
        Path(shutil.which(f"python{version}") or ""),
        Path(shutil.which("python3") or ""),
        Path(shutil.which("python") or ""),
    ):
        if candidate and candidate.exists() and is_expected_python_version(candidate, version):
            return candidate

    raise FileNotFoundError(f"Не удалось найти системный Python {version}.")


def resolve_build_python(target: BuildTarget) -> Path:
    override_value = os.environ.get(target.build_python_env_var, "").strip()
    if override_value:
        return Path(override_value)
    return resolve_managed_build_python(target)


def ensure_requirements_files_exist(target: BuildTarget) -> None:
    missing_files = [path for path in (target.runtime_requirements, target.build_requirements) if not path.exists()]
    if missing_files:
        missing = ", ".join(str(path) for path in missing_files)
        raise FileNotFoundError(f"Не найдены requirements-файлы для {target.label}: {missing}")


def ensure_build_venv(target: BuildTarget, build_python: Path) -> None:
    if build_python != resolve_managed_build_python(target):
        if not build_python.exists():
            raise FileNotFoundError(
                f"Interpreter from {target.build_python_env_var} not found: {build_python}"
            )
        version = read_python_version(build_python)
        print(f"== using {target.label} packaging interpreter: {build_python} (Python {version}) ==")
        if not version.startswith(target.preferred_python_version):
            raise RuntimeError(
                f"{target.label} override interpreter has wrong version: expected {target.preferred_python_version}, got {version}"
            )
        return

    system_python = resolve_system_python_for_version(target.preferred_python_version)
    recreate = False
    if build_python.exists():
        current_version = read_python_version(build_python)
        if not current_version.startswith(target.preferred_python_version):
            print(
                f"== recreating {target.label} packaging env: "
                f"found Python {current_version}, need {target.preferred_python_version} =="
            )
            recreate = True
    elif target.build_venv_dir.exists():
        recreate = True

    if recreate:
        remove_path(target.build_venv_dir)

    if build_python.exists():
        print(f"== reusing {target.label} packaging env: {build_python} (Python {read_python_version(build_python)}) ==")
        return

    print(f"== creating {target.label} packaging env with Python {target.preferred_python_version} ==")
    run([str(system_python), "-m", "venv", str(target.build_venv_dir)])
    created_version = read_python_version(build_python)
    if not created_version.startswith(target.preferred_python_version):
        raise RuntimeError(
            f"{target.label} packaging env created with wrong Python version: expected {target.preferred_python_version}, got {created_version}"
        )


def install_packaging_requirements(target: BuildTarget, build_python: Path) -> None:
    ensure_requirements_files_exist(target)
    print(
        f"== using {target.label} packaging interpreter: {build_python} "
        f"(Python {read_python_version(build_python)}) =="
    )
    run([str(build_python), "-m", "pip", "install", "--upgrade", "pip"])
    run([str(build_python), "-m", "pip", "install", "-r", str(target.runtime_requirements)])
    run([str(build_python), "-m", "pip", "install", "-r", str(target.build_requirements)])


def run_dependency_sanity_check(target: BuildTarget, build_python: Path) -> None:
    modules = ", ".join(repr(module) for module in target.sanity_modules)
    sanity_script = (
        "import importlib, sys\n"
        f"required = [{modules}]\n"
        "missing = []\n"
        "for name in required:\n"
        "    try:\n"
        "        importlib.import_module(name)\n"
        "    except Exception as exc:\n"
        "        missing.append(f'{name}: {exc}')\n"
        "if missing:\n"
        "    raise SystemExit('Missing critical packaging deps: ' + '; '.join(missing))\n"
        "print('Packaging deps OK')\n"
        "print('Packaging Python version:', sys.version.split()[0])\n"
    )
    print(f"== running {target.label} dependency sanity check ==")
    run([str(build_python), "-c", sanity_script])


def clean() -> None:
    print("== Cleaning old build artifacts ==")
    remove_path(BACKEND_DIST_DIR)
    remove_path(DESKTOP_DIST_DIR)
    remove_path(FINAL_DIR)
    remove_path(FINAL_ZIP_BASE.with_suffix(".zip"))
    remove_path(PYINSTALLER_BUILD_DIR)
    remove_path(DESKTOP_BACKEND_STAGING_DIR)
    remove_path(LEGACY_BUILD_VENV_DIR)


def build_ui() -> None:
    print("== Building UI ==")
    run(["npm.cmd", "run", "build"], cwd=UI_DIR)


def build_worker_target(target: BuildTarget) -> None:
    print(f"== Building {target.label} ==")
    ensure_dir(BACKEND_DIST_DIR)
    ensure_dir(target.pyinstaller_work_dir)
    ensure_executable_is_not_running(target.exe_path)

    build_python = resolve_build_python(target)
    ensure_build_venv(target, build_python)
    install_packaging_requirements(target, build_python)
    run_dependency_sanity_check(target, build_python)

    run(
        [
            str(build_python),
            "-m",
            "PyInstaller",
            str(target.entrypoint),
            "--noconfirm",
            "--distpath",
            str(BACKEND_DIST_DIR),
            "--workpath",
            str(target.pyinstaller_work_dir),
            "--specpath",
            str(target.pyinstaller_work_dir),
            "--paths",
            str(ROOT / "src"),
            "--name",
            target.dist_name,
        ],
        cwd=ROOT,
    )

    if not target.bundle_dir.is_dir():
        raise FileNotFoundError(f"Не найдена bundle directory для {target.label}: {target.bundle_dir}")
    if not target.exe_path.exists():
        raise FileNotFoundError(f"Не найден exe для {target.label}: {target.exe_path}")


def stage_backend_for_electron() -> None:
    print("== Staging backend workers for Electron ==")
    remove_path(DESKTOP_BACKEND_STAGING_DIR)
    ensure_dir(DESKTOP_BACKEND_STAGING_DIR)
    for target in BUILD_TARGETS:
        shutil.copytree(target.bundle_dir, target.desktop_stage_dir, dirs_exist_ok=True)
        if not target.desktop_stage_exe.exists():
            raise FileNotFoundError(
                f"Не найден staged exe для {target.label}: {target.desktop_stage_exe}"
            )


def build_desktop() -> None:
    print("== Building Electron desktop app ==")
    run(["npm.cmd", "install"], cwd=DESKTOP_DIR)
    run(["npm.cmd", "run", "build"], cwd=DESKTOP_DIR)


def create_final_package() -> Path:
    print("== Creating final tester package ==")
    ensure_dir(RELEASE_DIR)
    ensure_dir(FINAL_DIR)
    ensure_dir(FINAL_DIR / "data")
    ensure_dir(FINAL_DIR / "logs")
    ensure_dir(FINAL_DIR / "drivers")

    desktop_exe = find_desktop_exe()
    shutil.copy2(desktop_exe, FINAL_DIR / "EarLoop.exe")
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
    for target in BUILD_TARGETS:
        print(
            f"{target.label}: preferred Python {target.preferred_python_version}, "
            f"env var override {target.build_python_env_var}"
        )

    clean()
    build_ui()
    for target in BUILD_TARGETS:
        build_worker_target(target)
    stage_backend_for_electron()
    build_desktop()
    archive_path = create_final_package()

    print("\n== Done ==")
    for target in BUILD_TARGETS:
        print(f"{target.label} bundle: {target.bundle_dir}")
        print(f"{target.label} exe:    {target.exe_path}")
        print(f"{target.label} staged: {target.desktop_stage_exe}")
    print(f"Desktop dist: {DESKTOP_DIST_DIR}")
    print(f"Final folder: {FINAL_DIR}")
    print(f"Final zip:    {archive_path}")


if __name__ == "__main__":
    main()
