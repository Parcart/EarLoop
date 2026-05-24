from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

from .audio_bridge import (
    AudioDeviceChoice,
    NullAudioBridge,
    auto_detect_audio_devices,
    make_audio_bridge,
    print_devices,
    format_device_label,
    find_vb_cable_devices,
    format_device_row,
    get_default_audio_devices,
    is_vb_cable_capture_device,
)
from .config import load_config, save_config_template, persist_audio_selection
from .eval_strategies import EVAL_STRATEGY_PRESETS, evaluate_strategies
from .live_session import LivePersonalizationSession
from .profile_manager import ProfileManager
from .strategy import DEFAULT_STRATEGY, get_strategy, strategy_help
from .vendor_path import ensure_vendor_path
from .console import title, info, ok, warn, error, muted, key

ensure_vendor_path()
from personalization.contract_mapper import FREQS_23_DEFAULT


def _ask_int(prompt: str, default: int | None = None) -> int | None:
    suffix = f" [{default}]" if default is not None else ""
    while True:
        raw = input(f"{prompt}{suffix}: ").strip()
        if not raw and default is not None:
            return default
        if not raw:
            return None
        try:
            return int(raw)
        except ValueError:
            print("Введите числовой index устройства или Enter.")


def _print_selected_devices(capture: int | None, playback: int | None, cfg=None) -> None:
    allowed = getattr(getattr(cfg, "audio", None), "allowed_host_apis", None) if cfg is not None else None
    print("\n" + title("[audio] selected CLI devices"))
    print(f"  {key('capture/input')} : {format_device_label(capture, allowed_host_apis=allowed)}")
    print(f"  {key('playback/out ')} : {format_device_label(playback, allowed_host_apis=allowed)}")


def _confirm_or_edit_devices(
    *,
    capture: int | None,
    playback: int | None,
    cfg,
) -> tuple[bool, int | None, int | None]:
    """Ask the tester to confirm auto-detected devices and allow changing them.

    Returns: (enabled, capture, playback). enabled=False means dry-run.
    """
    confirm = bool(getattr(cfg.audio, "confirm_devices", True))
    if not confirm:
        return True, capture, playback

    while True:
        _print_selected_devices(capture, playback, cfg=cfg)
        print("\n" + warn("Важно для VB-Cable:"))
        print(f"  {key('Windows Output / Вывод Windows')} -> CABLE Input")
        print(f"  {key('CLI capture/input')}              -> CABLE Output")
        print(f"  {key('CLI playback/out')}               -> ваши наушники/колонки")
        print("  Не выбирайте CABLE Input как CLI playback/out, иначе обработанный звук уйдёт обратно в кабель.")
        print("\n" + info("Использовать эти CLI-устройства?"))
        print(f"  {key('Enter / y')}  - да, запустить аудиомост")
        print(f"  {key('c')}          - изменить CLI-устройства")
        print(f"  {key('l')}          - показать список устройств")
        print(f"  {key('d')}          - dry-run без реального аудио")
        answer = input("Ваш выбор [Y/c/l/d]: ").strip().lower()

        if answer in {"", "y", "yes", "д", "да"}:
            print(ok("[audio] CLI-устройства подтверждены. Дальше аудиомост будет запущен с этим routing."))
            sys.stdout.flush()
            return True, capture, playback

        if answer in {"d", "dry", "dry-run", "0"}:
            return False, capture, playback

        if answer in {"l", "list", "список"}:
            print_devices(allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None))
            continue

        if answer in {"c", "change", "n", "no", "нет", "изменить"}:
            print_devices(allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None))
            print("\nДля VB-Cable обычно capture = CABLE Output, playback = ваши наушники/колонки.")
            new_capture = _ask_int("Capture/input device index", capture)
            new_playback = _ask_int("Playback/output device index", playback)
            capture = new_capture if new_capture is not None else capture
            playback = new_playback if new_playback is not None else playback
            continue

        print("Не понял команду. Нажмите Enter для подтверждения или c для изменения.")




def _ask_audio_action(prompt: str, *, default: str = "continue") -> str:
    answer = input(prompt).strip().lower()
    if not answer:
        return default
    if answer in {"d", "dry", "dry-run", "0"}:
        return "dry"
    if answer in {"c", "change", "изменить", "n", "no", "нет"}:
        return "change"
    if answer in {"r", "refresh", "обновить", "проверить"}:
        return "refresh"
    if answer in {"x", "force", "ignore", "continue", "cont", "продолжить", "всё равно", "все равно"}:
        return "continue"
    print(f"Не понял команду: {answer!r}. Использую действие по умолчанию: {default}.")
    return default



def _safe_get_default_audio_devices(timeout_sec: float = 2.0) -> dict:
    """Read PortAudio default devices without blocking the whole CLI forever.

    On some Windows/PortAudio setups sd.default.device or device queries may hang
    right after the user switches the Windows output device. For tester builds the
    routing check is only advisory, so timeout should not prevent the session from
    starting.
    """
    import queue
    import threading

    result_queue: "queue.Queue[dict]" = queue.Queue(maxsize=1)

    def worker() -> None:
        try:
            result_queue.put(get_default_audio_devices(), block=False)
        except Exception as exc:
            try:
                result_queue.put({"input": None, "output": None, "error": str(exc)}, block=False)
            except Exception:
                pass

    thread = threading.Thread(target=worker, name="earloop-default-audio-check", daemon=True)
    thread.start()
    thread.join(timeout=float(timeout_sec))
    if thread.is_alive():
        return {
            "input": None,
            "output": None,
            "input_index": None,
            "output_index": None,
            "error": f"timeout after {timeout_sec:.1f}s while reading default audio device",
            "timeout": True,
        }
    try:
        return result_queue.get_nowait()
    except Exception:
        return {"input": None, "output": None, "error": "default audio check returned no data"}


def _print_vb_cable_warning(allowed_host_apis: list[str] | None) -> None:
    print("\n[audio] ВНИМАНИЕ: VB-Cable не найден среди аудиоустройств.")
    print("  Для стандартного тестового сценария нужен VB-Cable:")
    print("  1) Windows Output / Вывод звука  -> CABLE Input (VB-Audio Virtual Cable)")
    print("  2) CLI capture/input             -> CABLE Output")
    print("  3) CLI playback/output           -> ваши наушники/колонки")
    print("\n  Можно продолжить, если вы используете другой capture/input источник,")
    print("  но для тестеров лучше установить VB-Cable и перезапустить CLI.")
    if allowed_host_apis:
        print(f"  Сейчас устройства фильтруются по host API: {', '.join(allowed_host_apis)}")


def _post_device_setup_check(*, capture: int | None, playback: int | None, cfg) -> str:
    """Warn about missing VB-Cable and wrong Windows default output.

    Returns one of: continue, change, dry, refresh.
    """
    audio_cfg = getattr(cfg, "audio", None)
    allowed = getattr(audio_cfg, "allowed_host_apis", None)
    if not bool(getattr(audio_cfg, "warn_if_vb_cable_missing", True)) and str(getattr(audio_cfg, "expected_system_output", "auto")).lower() in {"", "none", "off", "false", "0"}:
        return "continue"

    vb = find_vb_cable_devices(allowed_host_apis=allowed)
    default = _safe_get_default_audio_devices(timeout_sec=1.0)
    current_out = default.get("output") if isinstance(default, dict) else None

    print("\n" + title("[audio] Windows/PortAudio default output check"))
    if default.get("error"):
        print(f"  не удалось определить: {default.get('error')}")
        if default.get("timeout"):
            print("  Это advisory-проверка. Чтобы не блокировать тест, можно продолжить с выбранными CLI-устройствами.")
            print("\nЧто сделать?")
            print("  Enter - продолжить")
            print("  c     - выбрать другие CLI-устройства")
            print("  d     - dry-run без реального аудио")
            action = _ask_audio_action("Ваш выбор [Enter/c/d]: ", default="continue")
            return action
    else:
        print(f"  {format_device_row(current_out)}")

    if bool(getattr(audio_cfg, "warn_if_vb_cable_missing", True)) and not bool(vb.get("installed")):
        _print_vb_cable_warning(allowed)
        print("\nЧто сделать?")
        print("  Enter - продолжить с выбранными устройствами")
        print("  c     - выбрать другие устройства")
        print("  d     - dry-run без реального аудио")
        print("  r     - обновить проверку после установки/переключения")
        return _ask_audio_action("Ваш выбор [Enter/c/d/r]: ", default="continue")

    expected_mode = str(getattr(audio_cfg, "expected_system_output", "auto") or "auto").strip().lower()
    if expected_mode in {"none", "off", "false", "0"}:
        return "continue"

    selected_capture_is_cable = is_vb_cable_capture_device(capture, allowed_host_apis=allowed)
    expected_row = None
    expected_reason = ""

    if expected_mode in {"vb", "vb_cable", "cable", "auto"} and (selected_capture_is_cable or expected_mode != "auto"):
        candidates = vb.get("playback_candidates") or []
        expected_row = candidates[0] if candidates else None
        expected_reason = "Для VB-Cable системный вывод Windows должен быть CABLE Input, иначе CLI не перехватит звук."
    elif expected_mode in {"selected", "selected_playback", "playback"}:
        from .audio_bridge import get_audio_device
        expected_row = get_audio_device(playback, allowed_host_apis=None)
        expected_reason = "По конфигу текущий системный вывод должен совпадать с выбранным playback/output."

    if expected_row is None:
        return "continue"

    current_idx = None if current_out is None else int(current_out.get("index", -999))
    expected_idx = int(expected_row.get("index", -1))
    if current_idx == expected_idx:
        return "continue"

    print("\n" + warn("[audio] Текущий системный вывод Windows может быть выбран не под VB-Cable."))
    print(f"  {expected_reason}")
    print("\n  Сейчас PortAudio сообщает default output:")
    print(f"    {format_device_row(current_out)}")
    print("  Ожидается для перехвата:")
    print(f"    {format_device_row(expected_row)}")
    print("\n" + warn("Важно:"))
    print("  Эта проверка информационная и не меняет настройки Windows.")
    print("  MME/PortAudio иногда показывает устаревший default output после переключения.")
    print("  Если при переключении ДО запуска моста пропадает звук — это может быть особенностью routing.")
    print("  Надёжный порядок для VB-Cable:")
    print("    1) оставить Windows Output на наушниках/колонках;")
    print("    2) запустить аудиомост;")
    print("    3) после сообщения bridge started переключить Windows Output на CABLE Input.")
    print("\nЧто сделать?")
    print(f"  {key('Enter')} - продолжить и запустить bridge")
    print(f"  {key('r')}     - обновить проверку ещё раз")
    print(f"  {key('c')}     - выбрать другие CLI-устройства")
    print(f"  {key('d')}     - dry-run без реального аудио")
    return _ask_audio_action("Ваш выбор [Enter/r/c/d]: ", default="continue")




def _print_final_audio_routing(*, capture: int | None, playback: int | None, cfg) -> None:
    """Print the exact routing that will be used right before AudioEngine start."""
    allowed = getattr(getattr(cfg, "audio", None), "allowed_host_apis", None)
    default = _safe_get_default_audio_devices(timeout_sec=1.0)
    current_out = default.get("output") if isinstance(default, dict) else None
    print("\n" + title("[audio] final routing before bridge start"))
    if isinstance(default, dict) and default.get("error"):
        print(f"  Windows Output / системный вывод : не удалось определить ({default.get('error')})")
    else:
        print(f"  Windows Output / системный вывод : {format_device_row(current_out)}")
    print(f"  CLI capture/input                : {format_device_label(capture, allowed_host_apis=allowed)}")
    print(f"  CLI playback/out                 : {format_device_label(playback, allowed_host_apis=allowed)}")
    if is_vb_cable_capture_device(capture, allowed_host_apis=allowed):
        print("  VB-Cable routing expected         : Windows Output=CABLE Input -> CLI capture=CABLE Output -> CLI playback=наушники/колонки")
    else:
        print("  Routing mode                      : обычный capture/input -> CLI playback/output")


def _ask_device_pair(capture: int | None, playback: int | None, cfg) -> tuple[int | None, int | None]:
    print_devices(allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None))
    print("\nДля VB-Cable обычно capture = CABLE Output, playback = ваши наушники/колонки.")
    new_capture = _ask_int("Capture/input device index", capture)
    new_playback = _ask_int("Playback/output device index", playback)
    capture = new_capture if new_capture is not None else capture
    playback = new_playback if new_playback is not None else playback
    return capture, playback

def _interactive_device_choice(cfg, no_audio: bool) -> tuple[bool, AudioDeviceChoice]:
    enabled = bool(cfg.audio.enabled) and not no_audio
    capture = cfg.audio.capture_device
    playback = cfg.audio.playback_device

    if enabled and (capture is None or playback is None) and bool(getattr(cfg.audio, "auto_detect_devices", True)):
        detected = auto_detect_audio_devices(
            sample_rate=int(cfg.sample_rate),
            channels=int(cfg.channels),
            allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None),
        )
        if detected is not None:
            capture = detected.capture_device if capture is None else capture
            playback = detected.playback_device if playback is None else playback
            print("[audio] auto-detected devices:")
            print(f"  capture/input : {format_device_label(capture, allowed_host_apis=getattr(cfg.audio, 'allowed_host_apis', None))}")
            print(f"  playback/out  : {format_device_label(playback, allowed_host_apis=getattr(cfg.audio, 'allowed_host_apis', None))}")

    if enabled and (capture is None or playback is None):
        print_devices(allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None))
        print("\nДля VB-Cable обычно capture = CABLE Output, playback = ваши наушники/колонки.")
        if capture is None:
            raw = input("Capture/input device index (или Enter для dry-run): ").strip()
            if not raw:
                enabled = False
            else:
                capture = int(raw)
        if enabled and playback is None:
            raw = input("Playback/output device index: ").strip()
            playback = int(raw)

    if enabled and capture is not None and playback is not None:
        while True:
            enabled, capture, playback = _confirm_or_edit_devices(
                capture=capture,
                playback=playback,
                cfg=cfg,
            )
            if not enabled:
                break
            action = _post_device_setup_check(capture=capture, playback=playback, cfg=cfg)
            if action == "continue":
                break
            if action == "dry":
                enabled = False
                break
            if action == "change":
                capture, playback = _ask_device_pair(capture, playback, cfg)
                continue
            if action == "refresh":
                continue
            break

    return enabled, AudioDeviceChoice(
        capture_device=capture,
        playback_device=playback,
        sample_rate=int(cfg.sample_rate),
        channels=int(cfg.channels),
        blocksize=int(cfg.blocksize),
        latency=str(cfg.audio.latency),
        allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None),
    )


def _select_profile_start_z(data_root: Path) -> np.ndarray | None:
    """Interactive saved-profile picker for starting a new session."""
    pm = ProfileManager(data_root)
    profiles = pm.print_profiles()
    if not profiles:
        return None
    raw = input("\nВыберите профиль как стартовую точку или Enter для отмены: ").strip()
    if not raw:
        return None
    try:
        idx = int(raw)
    except ValueError:
        print("Нужно ввести номер профиля.")
        return None
    if idx < 1 or idx > len(profiles):
        print("Нет такого профиля.")
        return None
    profile = pm.load(profiles[idx - 1])
    print(f"Следующая сессия начнётся от профиля: {profile.name}")
    return np.asarray(profile.z_contract, dtype=np.float64)


def _make_session_audio(enabled: bool, choice: AudioDeviceChoice, preamp_db: float):
    try:
        return make_audio_bridge(enabled=enabled, choice=choice, preamp_db=float(preamp_db))
    except Exception as exc:
        print(f"[audio] failed to start real bridge: {exc}")
        print("[audio] fallback to dry-run mode")
        return NullAudioBridge()


def _pre_bridge_headphones_prompt(*, enabled: bool, choice: AudioDeviceChoice, cfg, session_index: int) -> None:
    """Prompt before starting the bridge for stable VB-Cable routing.

    Empirically on some Windows/MME setups the bridge starts reliably only if
    Windows Output is still the real headphones/speakers at the moment of engine
    initialization. After [audio] bridge started, LiveSession prompts the tester
    to switch Windows Output to CABLE Input.
    """
    if not enabled:
        return
    audio_cfg = getattr(cfg, "audio", None)
    if not bool(getattr(audio_cfg, "pre_bridge_headphones_prompt", True)):
        return
    capture = getattr(choice, "capture_device", None)
    allowed = getattr(choice, "allowed_host_apis", None) or getattr(audio_cfg, "allowed_host_apis", None)
    if not is_vb_cable_capture_device(capture, allowed_host_apis=allowed):
        return

    if session_index <= 1:
        print("\n" + title("[audio] перед запуском аудиомоста"))
    else:
        print("\n" + title("[audio] новая сессия: подготовка routing перед перезапуском моста"))
    print("Чтобы VB-Cable/MME стартовал стабильно:")
    print(f"  1) {key('ПЕРЕД стартом моста')} поставьте Windows Output на реальные наушники/колонки.")
    print(f"  2) CLI capture/input остаётся: {format_device_label(capture, allowed_host_apis=allowed)}")
    print("  3) После сообщения [audio] bridge started CLI попросит переключить Windows Output на CABLE Input.")
    print(warn("Не запускайте мост, когда Windows Output уже CABLE Input, если на вашей системе из-за этого пропадает звук."))
    input("Переключите Windows Output на наушники/колонки и нажмите Enter для запуска аудиомоста...")


def _post_session_menu(*, summary, session: LivePersonalizationSession, data_root: Path) -> tuple[bool, np.ndarray | None]:
    """Return (start_next_session, next_initial_z)."""
    print("\nЧто дальше?")
    print("[1] Начать новую сессию с нуля")
    print("[2] Продолжить настройку от текущего итогового состояния")
    print("[3] Начать новую сессию от сохранённого профиля")
    print("[0] Выйти")

    default = "0"
    try:
        raw = input(f"Ваш выбор [{default}]: ").strip().lower() or default
    except EOFError:
        raw = default

    if raw == "1":
        return True, None
    if raw == "2":
        z = np.asarray(session.state.z_mean, dtype=np.float64).copy()
        print("Следующая сессия начнётся от текущего итогового z_contract.")
        return True, z
    if raw == "3":
        z = _select_profile_start_z(data_root)
        if z is None:
            return False, None
        return True, z
    return False, None


def cmd_run(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    strategy = get_strategy(args.strategy or cfg.default_strategy)
    print("\n" + title(f"Strategy: {strategy.name}"))
    print(strategy.description)
    print(info(f"Mapper mode: {cfg.mapper.mode}, path: {cfg.mapper.model_path}"))
    print(info(f"Audio preamp: {float(cfg.preamp_db):+.1f} dB"))
    data_root = Path(args.data_dir or cfg.data_dir).resolve()
    data_root.mkdir(parents=True, exist_ok=True)

    enabled, choice = _interactive_device_choice(cfg, no_audio=args.no_audio)
    if enabled and choice.capture_device is not None and choice.playback_device is not None:
        _print_final_audio_routing(capture=choice.capture_device, playback=choice.playback_device, cfg=cfg)
        cfg.audio.capture_device = int(choice.capture_device)
        cfg.audio.playback_device = int(choice.playback_device)
        cfg.audio.auto_detect_devices = False
        try:
            persist_audio_selection(
                args.config,
                capture_device=cfg.audio.capture_device,
                playback_device=cfg.audio.playback_device,
                allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None),
            )
            print(ok(f"[audio] device choice saved to {args.config}"))
        except Exception as exc:
            print(f"[audio] warning: cannot save device choice: {exc}")

    initial_z = None
    if args.start_profile:
        pm = ProfileManager(data_root)
        profile = pm.load(args.start_profile)
        initial_z = np.asarray(profile.z_contract, dtype=np.float64)
        print(f"Starting from profile: {profile.name}")

    # Keep one audio bridge instance for the whole CLI run.
    # This mirrors the desktop app more closely: the bridge stays alive, while
    # personalization sessions are restarted on top of the same locked audio route.
    # It also avoids the Windows/MME + VB-Cable bug where recreating the stream
    # while Windows Output is already CABLE Input may start the bridge in silence.
    _pre_bridge_headphones_prompt(enabled=enabled, choice=choice, cfg=cfg, session_index=1)
    audio = _make_session_audio(enabled=enabled, choice=choice, preamp_db=float(cfg.preamp_db))

    session_index = 1
    try:
        while True:
            print("\n" + title(f"=== Personalization session #{session_index} ==="))
            if initial_z is None:
                print("Старт: neutral / fresh preference state")
            else:
                print("Старт: от заданного z_contract / сохранённого профиля")

            if session_index > 1:
                print(ok("[audio] используется уже запущенный аудиомост; переключать Windows Output обратно на наушники не нужно"))

            session = LivePersonalizationSession(
                config=cfg,
                strategy=strategy,
                audio=audio,
                data_root=data_root,
                initial_z=initial_z,
            )
            summary = session.run()
            print("\nSession summary:")
            print(summary)

            should_continue, next_initial_z = _post_session_menu(summary=summary, session=session, data_root=data_root)
            if not should_continue:
                break
            initial_z = next_initial_z
            session_index += 1
    finally:
        try:
            audio.apply_neutral()
            audio.stop()
        except Exception as exc:
            print(f"[audio] warning: cannot stop bridge on exit: {exc}")


def cmd_profiles(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_root = Path(args.data_dir or cfg.data_dir).resolve()
    pm = ProfileManager(data_root)
    profiles = pm.print_profiles()
    if not profiles:
        return
    raw = input("\nВыберите профиль для действий или Enter для выхода: ").strip()
    if not raw:
        return
    idx = int(raw)
    if idx < 1 or idx > len(profiles):
        return
    profile = pm.load(profiles[idx - 1])
    print("\nz_contract:")
    print(pm.feature_table(profile.z_contract))
    action = input("[a] apply profile via audio bridge, [p] print EQ, [q] quit: ").strip().lower()
    if action == "p":
        for f, g in zip(profile.freqs_23, profile.eq23_db):
            print(f"{float(f):>8.1f} Hz : {float(g): .3f} dB")
        return
    if action == "a":
        enabled, choice = _interactive_device_choice(cfg, no_audio=False)
        audio = make_audio_bridge(enabled=enabled, choice=choice, preamp_db=float(cfg.preamp_db))
        try:
            audio.start()
            audio.apply_eq(profile.name, profile.freqs_23, profile.eq23_db, preamp_db=float(cfg.preamp_db))
            input("Профиль включен. Нажмите Enter для neutral/stop...")
            audio.apply_neutral()
        finally:
            audio.stop()


def cmd_devices(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    print_devices(allowed_host_apis=getattr(cfg.audio, "allowed_host_apis", None))


def cmd_strategies(args: argparse.Namespace) -> None:
    print("Available strategies:")
    print(strategy_help())


def cmd_eval_strategies(args: argparse.Namespace) -> None:
    cfg = load_config(args.config)
    data_root = Path(args.data_dir or cfg.data_dir).resolve()
    out_dir = data_root / "evaluations" / args.name
    strategies = args.strategies or list(EVAL_STRATEGY_PRESETS.keys())
    path = evaluate_strategies(
        out_dir=out_dir,
        strategies=strategies,
        n_users=int(args.n_users),
        budget=int(args.budget),
        seed=int(args.seed),
    )
    print(f"Saved strategy comparison: {path}")


def cmd_init_config(args: argparse.Namespace) -> None:
    save_config_template(args.output)
    print(f"Config template saved: {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="earloop-personalization-cli")
    parser.add_argument("--config", default="config.json", help="Path to config.json")
    parser.add_argument("--data-dir", default=None, help="Override runtime data dir")
    sub = parser.add_subparsers(dest="cmd", required=False)

    p = sub.add_parser("run", help="Run interactive real-user personalization session")
    p.add_argument("--strategy", default=None, help=f"Strategy name, default={DEFAULT_STRATEGY}")
    p.add_argument("--no-audio", action="store_true", help="Disable real audio bridge / dry-run")
    p.add_argument("--start-profile", default=None, help="Start session from saved profile JSON")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("profiles", help="List/apply saved profiles")
    p.set_defaults(func=cmd_profiles)

    p = sub.add_parser("devices", help="List audio devices")
    p.set_defaults(func=cmd_devices)

    p = sub.add_parser("strategies", help="Show available final strategies")
    p.set_defaults(func=cmd_strategies)

    p = sub.add_parser("eval-strategies", help="Offline synthetic comparison of final strategies")
    p.add_argument("--name", default="final_strategy_eval", help="Evaluation output folder name")
    p.add_argument("--strategies", nargs="*", default=None, help="Strategies to compare")
    p.add_argument("--n-users", type=int, default=80)
    p.add_argument("--budget", type=int, default=24)
    p.add_argument("--seed", type=int, default=42)
    p.set_defaults(func=cmd_eval_strategies)

    p = sub.add_parser("init-config", help="Write config template")
    p.add_argument("--output", default="config.json")
    p.set_defaults(func=cmd_init_config)

    return parser


def main(argv: list[str] | None = None) -> None:
    if argv is None:
        argv = sys.argv[1:]

    # Double-click exe / PyCharm without parameters: start the tester session.
    # This prevents the console from opening and immediately closing with "cmd required".
    if not argv:
        argv = ["run"]

    parser = build_parser()
    args = parser.parse_args(argv)
    if not hasattr(args, "func"):
        args = parser.parse_args(["run"])
    args.func(args)
