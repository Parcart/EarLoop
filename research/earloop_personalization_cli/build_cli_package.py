from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DIST_DIR = ROOT / "dist"
BUILD_DIR = ROOT / "build" / "pyinstaller"
RELEASE_DIR = ROOT / "release"
FINAL_DIR = RELEASE_DIR / "EarLoop-Personalization-CLI"
FINAL_ZIP_BASE = RELEASE_DIR / "EarLoop-Personalization-CLI"
VENV_DIR_DEFAULT = ROOT / ".venv-build-cli"

README_TEXT = """EarLoop Personalization CLI — тестовый билд

Назначение:
- пройти A/B-сессию персонализации EQ;
- протестировать Directional Feedback;
- сохранить профиль;
- отправить исследователю архив логов.

Как запустить:
1. Распакуйте архив.
2. Установите VB-Cable, если тестируете системный звук через виртуальный аудиомост.
3. Запустите EarLoopPersonalizationCLI.exe. Команду run писать не обязательно: без аргументов запускается обычная сессия.
4. Выберите capture/input устройство: обычно CABLE Output.
5. Выберите playback/output устройство: ваши наушники/колонки.
6. Следуйте командам в консоли.

Основные команды в A/B-сессии:
1 — включить A
2 — включить B
a — выбрать A
b — выбрать B
f — Directional Feedback, если оба варианта плохие
p — включить текущий лучший профиль
n — neutral EQ
s — сохранить и завершить
q — выйти без сохранения

После завершения отправьте архив из runtime_data/exports/.
"""


@dataclass(frozen=True)
class PythonCandidate:
    path: Path
    source: str


def run(cmd: list[str], cwd: Path | None = None, *, check: bool = True, capture: bool = False) -> subprocess.CompletedProcess:
    printable = " ".join(str(x) for x in cmd)
    print(f">>> {printable}")
    return subprocess.run(
        cmd,
        cwd=cwd or ROOT,
        check=check,
        capture_output=capture,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )


def remove(path: Path) -> None:
    if path.is_dir():
        shutil.rmtree(path, ignore_errors=True)
    elif path.exists():
        path.unlink()


def read_python_version(py: Path) -> str:
    result = run([str(py), "-c", "import sys; print(sys.version.split()[0])"], check=True, capture=True)
    return (result.stdout or "").strip()


def is_inside_venv(py: Path) -> bool:
    parts = {part.lower() for part in py.parts}
    return any(part.startswith(".venv") or part in {"venv", "env"} for part in parts)


def venv_python(venv_dir: Path) -> Path:
    return venv_dir / ("Scripts/python.exe" if sys.platform == "win32" else "bin/python")


def resolve_py_launcher(version: str) -> Path | None:
    if sys.platform != "win32":
        return None
    launcher = shutil.which("py")
    if not launcher:
        return None
    result = subprocess.run(
        [launcher, f"-{version}", "-c", "import sys; print(sys.executable)"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="ignore",
    )
    if result.returncode == 0 and result.stdout.strip():
        return Path(result.stdout.strip())
    return None


def find_system_python(preferred_versions: tuple[str, ...]) -> list[PythonCandidate]:
    candidates: list[PythonCandidate] = []

    # Explicit environment override has highest priority.
    env_override = os.environ.get("EARLOOP_CLI_BUILD_PYTHON", "").strip()
    if env_override:
        candidates.append(PythonCandidate(Path(env_override), "EARLOOP_CLI_BUILD_PYTHON"))

    # If this script is run from a project venv, sys._base_executable is usually the real system python.
    base_executable = getattr(sys, "_base_executable", "")
    if base_executable:
        base_path = Path(base_executable)
        if base_path.exists() and base_path != Path(sys.executable):
            candidates.append(PythonCandidate(base_path, "sys._base_executable"))

    # Windows launcher: py -3.11, py -3.12, etc.
    for version in preferred_versions:
        py = resolve_py_launcher(version)
        if py is not None:
            candidates.append(PythonCandidate(py, f"py -{version}"))

    # PATH lookup fallback.
    for name in [*(f"python{v}" for v in preferred_versions), "python3", "python"]:
        resolved = shutil.which(name)
        if resolved:
            candidates.append(PythonCandidate(Path(resolved), f"PATH:{name}"))

    # Deduplicate by resolved path.
    unique: list[PythonCandidate] = []
    seen: set[str] = set()
    for item in candidates:
        try:
            key = str(item.path.resolve()).lower()
        except Exception:
            key = str(item.path).lower()
        if key in seen:
            continue
        seen.add(key)
        unique.append(item)
    return unique


def create_build_venv(venv_dir: Path, python_override: str | None, recreate: bool, allow_current_python: bool) -> Path:
    py = venv_python(venv_dir)
    if recreate:
        remove(venv_dir)
    if py.exists():
        print(f"== Reusing build venv: {py} (Python {read_python_version(py)}) ==")
        return py

    if python_override:
        candidates = [PythonCandidate(Path(python_override), "--python")]
    else:
        candidates = find_system_python(("3.11", "3.12", "3.10"))

    if allow_current_python:
        candidates.append(PythonCandidate(Path(sys.executable), "current sys.executable"))

    errors: list[str] = []
    for candidate in candidates:
        base_py = candidate.path
        if not base_py.exists():
            errors.append(f"{candidate.source}: not found: {base_py}")
            continue
        if not allow_current_python and base_py == Path(sys.executable) and is_inside_venv(base_py):
            errors.append(f"{candidate.source}: skipped project venv python: {base_py}")
            continue
        try:
            version = read_python_version(base_py)
        except Exception as exc:
            errors.append(f"{candidate.source}: cannot read version: {exc}")
            continue

        print(f"== Creating isolated build venv with {candidate.source}: {base_py} (Python {version}) ==")
        remove(venv_dir)
        result = run([str(base_py), "-m", "venv", str(venv_dir)], check=False)
        if result.returncode == 0 and py.exists():
            print(f"== Build venv created: {py} ==")
            return py
        errors.append(f"{candidate.source}: venv creation failed with code {result.returncode}")

    message = "\n".join(errors[-12:])
    raise RuntimeError(
        "Не удалось создать отдельную build-venv.\n"
        "Важно: сборка специально НЕ использует большой .venv проекта, чтобы exe не раздувался.\n"
        "Попробуй указать системный Python явно, например:\n"
        "  py -3.11 build_cli_package.py --recreate-venv\n"
        "или:\n"
        "  python build_cli_package.py --python C:\\Users\\makcc\\AppData\\Local\\Programs\\Python\\Python311\\python.exe --recreate-venv\n\n"
        f"Диагностика:\n{message}"
    )


def ensure_tools(py: Path) -> None:
    print(f"== Installing minimal build/runtime deps into isolated env: {py} ==")
    run([str(py), "-m", "pip", "install", "--upgrade", "pip", "setuptools", "wheel"])
    run([str(py), "-m", "pip", "install", "-r", str(ROOT / "requirements.txt")])
    # requirements.txt already includes pyinstaller, but this keeps intent explicit.
    run([str(py), "-m", "pip", "install", "pyinstaller"])


def build(args: argparse.Namespace) -> Path:
    remove(DIST_DIR)
    remove(BUILD_DIR)
    remove(FINAL_DIR)
    remove(FINAL_ZIP_BASE.with_suffix(".zip"))
    RELEASE_DIR.mkdir(parents=True, exist_ok=True)

    venv_dir = Path(args.venv_dir) if args.venv_dir else VENV_DIR_DEFAULT
    py = create_build_venv(
        venv_dir=venv_dir,
        python_override=args.python,
        recreate=args.recreate_venv,
        allow_current_python=args.allow_current_python,
    )
    ensure_tools(py)
    print(f"== Build Python: {py} (Python {read_python_version(py)}) ==")

    pyinstaller_cmd = [
        str(py), "-m", "PyInstaller",
        str(ROOT / "main.py"),
        "--noconfirm",
        "--clean",
        "--console",
        "--name", "EarLoopPersonalizationCLI",
        "--distpath", str(DIST_DIR),
        "--workpath", str(BUILD_DIR),
        "--specpath", str(BUILD_DIR),
        "--add-data", f"{ROOT / 'vendor'}{os.pathsep}vendor",
        "--add-data", f"{ROOT / 'config.example.json'}{os.pathsep}.",
        "--add-data", f"{ROOT / 'models'}{os.pathsep}models" if (ROOT / "models").exists() else f"{ROOT / 'config.example.json'}{os.pathsep}.",
        # Keep bundle smaller and avoid accidental capture from developer envs.
        "--exclude-module", "matplotlib",
        "--exclude-module", "notebook",
        "--exclude-module", "jupyter",
        "--exclude-module", "IPython",
        "--exclude-module", "torch",
        "--exclude-module", "tensorflow",
        "--exclude-module", "sklearn",
        # Audio backend dependencies can use dynamic imports; keep them explicit.
        "--hidden-import", "scipy.signal",
        "--hidden-import", "scipy.fft",
        "--hidden-import", "scipy.interpolate",
        "--hidden-import", "sounddevice",
        "--hidden-import", "soundfile",
        "--collect-submodules", "scipy.signal",
    ]
    run(pyinstaller_cmd, cwd=ROOT)

    bundle = DIST_DIR / "EarLoopPersonalizationCLI"
    if not bundle.exists():
        raise FileNotFoundError(bundle)

    shutil.copytree(bundle, FINAL_DIR, dirs_exist_ok=True)
    shutil.copy2(ROOT / "config.example.json", FINAL_DIR / "config.json")
    if (ROOT / "models").exists():
        shutil.copytree(ROOT / "models", FINAL_DIR / "models", dirs_exist_ok=True)
    (FINAL_DIR / "README.txt").write_text(README_TEXT, encoding="utf-8")
    (FINAL_DIR / "runtime_data").mkdir(exist_ok=True)

    meta = {
        "build_python": str(py),
        "python_version": read_python_version(py),
        "isolated_build_venv": str(venv_dir),
        "uses_project_venv": False,
    }
    (FINAL_DIR / "build_info.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    archive = shutil.make_archive(str(FINAL_ZIP_BASE), "zip", root_dir=FINAL_DIR.parent, base_dir=FINAL_DIR.name)
    return Path(archive)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build isolated EarLoop Personalization CLI tester package")
    parser.add_argument(
        "--python",
        default=None,
        help="Явный путь к системному python.exe для создания маленькой build-venv. Не указывай большой project .venv.",
    )
    parser.add_argument(
        "--venv-dir",
        default=None,
        help="Папка для build-venv. По умолчанию: .venv-build-cli рядом со сборщиком.",
    )
    parser.add_argument(
        "--recreate-venv",
        action="store_true",
        help="Удалить и создать .venv-build-cli заново.",
    )
    parser.add_argument(
        "--allow-current-python",
        action="store_true",
        help="Аварийный режим: разрешить использовать текущий Python. По умолчанию выключено, чтобы не раздувать exe большим env.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    archive = build(args)
    print(f"Done: {archive}")
