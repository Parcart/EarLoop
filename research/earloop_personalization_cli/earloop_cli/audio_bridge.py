from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any

import numpy as np

from .vendor_path import ensure_vendor_path
ensure_vendor_path()


def _normalize_hostapis(allowed_host_apis: list[str] | tuple[str, ...] | None) -> list[str]:
    if not allowed_host_apis:
        return []
    return [str(x).strip().lower() for x in allowed_host_apis if str(x).strip()]


def _hostapi_allowed(hostapi: str, allowed_host_apis: list[str] | tuple[str, ...] | None) -> bool:
    allowed = _normalize_hostapis(allowed_host_apis)
    if not allowed:
        return True
    host = str(hostapi).strip().lower()
    return any(token == host or token in host for token in allowed)


@dataclass
class AudioDeviceChoice:
    capture_device: int | None
    playback_device: int | None
    sample_rate: int
    channels: int
    blocksize: int
    latency: str = "low"
    allowed_host_apis: list[str] | None = None


class NullAudioBridge:
    """No-audio backend for dry runs and non-Windows/non-audio environments."""

    def __init__(self) -> None:
        self.active_label = "neutral"
        self.running = False
        self.last_eq = None

    def list_devices(self) -> list[dict[str, Any]]:
        return []

    def start(self) -> None:
        self.running = True
        print("[audio] dry-run mode: real audio bridge is disabled")

    def stop(self) -> None:
        self.running = False

    def apply_eq(self, label: str, freqs_hz, gains_db, preamp_db: float = 0.0) -> None:
        self.active_label = label
        self.last_eq = np.asarray(gains_db, dtype=np.float32).copy()
        print(f"[audio] dry-run apply {label}: max_abs={float(np.max(np.abs(self.last_eq))):.2f} dB")

    def apply_neutral(self) -> None:
        self.active_label = "neutral"
        self.last_eq = None
        print("[audio] dry-run neutral EQ")

    def health(self) -> dict[str, Any]:
        return {"backend": "null", "running": self.running, "active_label": self.active_label}


class RealtimeAudioBridge:
    """Direct real-time bridge using uploaded backend earloop.audio classes.

    Capture device -> EqualizerProcessor -> playback device.
    Typical tester setup: VB-Cable/CABLE Output as capture device and headphones/speakers as playback device.
    """

    def __init__(self, choice: AudioDeviceChoice, preamp_db: float = 0.0) -> None:
        from earloop.audio.engine import AudioEngine
        from earloop.audio.processors import EqualizerProcessor

        if choice.capture_device is None or choice.playback_device is None:
            raise ValueError("capture_device and playback_device must be selected for real audio mode")

        self.choice = choice
        self.preamp_db = float(preamp_db)
        self.processor = EqualizerProcessor(samplerate=choice.sample_rate, channels=choice.channels)
        self.engine = AudioEngine(
            capture_device=int(choice.capture_device),
            playback_device=int(choice.playback_device),
            samplerate=int(choice.sample_rate),
            channels=int(choice.channels),
            blocksize=int(choice.blocksize),
            latency=str(choice.latency),
            processor=self.processor,
        )
        self.active_label = "neutral"
        self.running = False

    @staticmethod
    def list_devices(allowed_host_apis: list[str] | tuple[str, ...] | None = None) -> list[dict[str, Any]]:
        import sounddevice as sd
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        rows: list[dict[str, Any]] = []
        for i, dev in enumerate(devices):
            hostapi = str(hostapis[int(dev["hostapi"])] ["name"])
            if not _hostapi_allowed(hostapi, allowed_host_apis):
                continue
            rows.append({
                "index": int(i),
                "name": str(dev.get("name", "")),
                "hostapi": hostapi,
                "max_input_channels": int(dev.get("max_input_channels", 0)),
                "max_output_channels": int(dev.get("max_output_channels", 0)),
                "default_samplerate": int(float(dev.get("default_samplerate", 0))),
            })
        return rows

    def start(self) -> None:
        if self.running:
            print("[audio] bridge already running; reusing locked route")
            return
        self.engine.start()
        self.running = True
        time.sleep(0.2)
        print("[audio] bridge started")

    def stop(self) -> None:
        if not self.running:
            return
        try:
            self.engine.stop()
        finally:
            self.running = False
        print("[audio] bridge stopped")

    def apply_eq(self, label: str, freqs_hz, gains_db, preamp_db: float | None = None) -> None:
        from earloop.audio.processors import EqProfile
        profile = EqProfile(
            profile_id=abs(hash(label)) % 1_000_000,
            freqs_hz=np.asarray(freqs_hz, dtype=np.float64),
            gains_db=np.asarray(gains_db, dtype=np.float64),
            preamp_db=self.preamp_db if preamp_db is None else float(preamp_db),
            name=label,
        )
        self.processor.set_profile(profile)
        self.active_label = label
        print(f"[audio] applied {label}: max_abs={float(np.max(np.abs(profile.gains_db))):.2f} dB")

    def apply_neutral(self) -> None:
        self.processor.clear_profile()
        self.active_label = "neutral"
        print("[audio] neutral / passthrough")

    def health(self) -> dict[str, Any]:
        return {"backend": "realtime", "running": self.running, "active_label": self.active_label, **self.engine.get_debug_snapshot()}


def auto_detect_audio_devices(sample_rate: int = 48000, channels: int = 2, allowed_host_apis: list[str] | None = None) -> AudioDeviceChoice | None:
    """Best-effort device detection for tester builds.

    Capture: prefer VB-Cable/CABLE Output input-like device.
    Playback: prefer system default output.
    Returns None when detection is not reliable.
    """
    try:
        import sounddevice as sd
        rows = RealtimeAudioBridge.list_devices(allowed_host_apis=allowed_host_apis)
        _default_in, default_out = sd.default.device
    except Exception:
        return None

    capture = None
    cable_tokens = ("cable output", "vb-audio", "vb cable", "cable-out")
    for row in rows:
        name = str(row.get("name", "")).lower()
        if int(row.get("max_input_channels", 0)) > 0 and any(tok in name for tok in cable_tokens):
            capture = int(row["index"])
            break

    playback = None
    try:
        if default_out is not None and int(default_out) >= 0 and any(int(r["index"]) == int(default_out) for r in rows):
            playback = int(default_out)
    except Exception:
        playback = None
    if playback is None:
        for row in rows:
            if int(row.get("max_output_channels", 0)) > 0:
                playback = int(row["index"])
                break

    if capture is None or playback is None:
        return None

    return AudioDeviceChoice(
        capture_device=capture,
        playback_device=playback,
        sample_rate=int(sample_rate),
        channels=int(channels),
        blocksize=1024,
        latency="low",
        allowed_host_apis=list(allowed_host_apis) if allowed_host_apis is not None else None,
    )


def get_audio_device(index: int | None, allowed_host_apis: list[str] | None = None) -> dict[str, Any] | None:
    if index is None:
        return None
    try:
        rows = RealtimeAudioBridge.list_devices(allowed_host_apis=allowed_host_apis)
    except Exception:
        return None
    for row in rows:
        try:
            if int(row.get("index", -1)) == int(index):
                return row
        except Exception:
            continue
    # If a saved config points to a non-filtered host API, still show its real name.
    if allowed_host_apis:
        try:
            rows = RealtimeAudioBridge.list_devices(allowed_host_apis=None)
            for row in rows:
                if int(row.get("index", -1)) == int(index):
                    return row
        except Exception:
            pass
    return None


def format_device_label(index: int | None, allowed_host_apis: list[str] | None = None) -> str:
    if index is None:
        return "not selected"
    row = get_audio_device(index, allowed_host_apis=allowed_host_apis)
    if row is None:
        return f"{index} | unknown device"
    return format_device_row(row)





def _cable_name_tokens() -> tuple[str, ...]:
    return (
        "vb-audio",
        "vb cable",
        "virtual cable",
        "cable input",
        "cable output",
        "cable-input",
        "cable-output",
    )


def _row_name(row: dict[str, Any] | None) -> str:
    return str((row or {}).get("name", "")).strip().lower()


def _looks_like_vb_cable(row: dict[str, Any] | None) -> bool:
    name = _row_name(row)
    if not name:
        return False
    return any(tok in name for tok in _cable_name_tokens())


def find_vb_cable_devices(allowed_host_apis: list[str] | None = None) -> dict[str, Any]:
    """Return detected VB-Cable-like endpoints.

    In the usual Windows setup:
    - CABLE Input is a playback/output endpoint and should be selected as the
      system default output, so system sound goes into the virtual cable.
    - CABLE Output is a recording/input endpoint and should be selected as the
      CLI capture device.
    """
    try:
        rows = RealtimeAudioBridge.list_devices(allowed_host_apis=allowed_host_apis)
    except Exception:
        rows = []
    # Fallback: if MME-only filtering hides the cable, inspect all devices for a warning.
    if allowed_host_apis and not any(_looks_like_vb_cable(row) for row in rows):
        try:
            rows_all = RealtimeAudioBridge.list_devices(allowed_host_apis=None)
        except Exception:
            rows_all = []
    else:
        rows_all = rows

    inputs: list[dict[str, Any]] = []
    outputs: list[dict[str, Any]] = []
    for row in rows_all:
        if not _looks_like_vb_cable(row):
            continue
        name = _row_name(row)
        is_capture_name = "cable output" in name or "cable-output" in name
        is_playback_name = "cable input" in name or "cable-input" in name
        if int(row.get("max_input_channels", 0)) > 0 and (is_capture_name or not is_playback_name):
            inputs.append(row)
        if int(row.get("max_output_channels", 0)) > 0 and (is_playback_name or not is_capture_name):
            outputs.append(row)

    return {
        "installed": bool(inputs or outputs),
        "capture_candidates": inputs,
        "playback_candidates": outputs,
    }


def is_vb_cable_capture_device(index: int | None, allowed_host_apis: list[str] | None = None) -> bool:
    row = get_audio_device(index, allowed_host_apis=allowed_host_apis)
    if row is None:
        return False
    return _looks_like_vb_cable(row) and int(row.get("max_input_channels", 0)) > 0


def is_vb_cable_playback_device(index: int | None, allowed_host_apis: list[str] | None = None) -> bool:
    row = get_audio_device(index, allowed_host_apis=allowed_host_apis)
    if row is None:
        return False
    return _looks_like_vb_cable(row) and int(row.get("max_output_channels", 0)) > 0


def get_default_audio_devices() -> dict[str, Any]:
    """Best-effort current PortAudio default input/output devices.

    This usually reflects the active Windows default device as seen by PortAudio.
    If unavailable, returns indices as None and an error string.
    """
    try:
        import sounddevice as sd
        default_in, default_out = sd.default.device
    except Exception as exc:
        return {"input": None, "output": None, "error": str(exc)}

    def _safe_idx(value):
        try:
            if value is None or int(value) < 0:
                return None
            return int(value)
        except Exception:
            return None

    in_idx = _safe_idx(default_in)
    out_idx = _safe_idx(default_out)
    return {
        "input": get_audio_device(in_idx, allowed_host_apis=None),
        "output": get_audio_device(out_idx, allowed_host_apis=None),
        "input_index": in_idx,
        "output_index": out_idx,
        "error": None,
    }


def format_device_row(row: dict[str, Any] | None) -> str:
    """Format one PortAudio device row for confirmation prompts.

    This function must be leaf-only. A previous patch accidentally made it call
    itself, which caused RecursionError while printing selected devices.
    """
    if row is None:
        return "not selected"

    def _get_int(key: str, default: int = 0) -> int:
        try:
            return int(row.get(key, default))
        except Exception:
            return default

    index = _get_int("index", -1)
    name = str(row.get("name", "unknown device"))
    hostapi = str(row.get("hostapi", "unknown hostapi"))
    ins = _get_int("max_input_channels", 0)
    outs = _get_int("max_output_channels", 0)
    try:
        sr = int(float(row.get("default_samplerate", 0)))
    except Exception:
        sr = 0

    sr_part = f", sr={sr}" if sr > 0 else ""
    return f"{index} | {name} [{hostapi}, in={ins}, out={outs}{sr_part}]"

def print_devices(allowed_host_apis: list[str] | None = None) -> None:
    try:
        rows = RealtimeAudioBridge.list_devices(allowed_host_apis=allowed_host_apis)
    except Exception as exc:
        print(f"[audio] cannot list devices: {exc}")
        return
    if allowed_host_apis:
        print(f"\nAvailable audio devices (host APIs: {', '.join(allowed_host_apis)}):")
    else:
        print("\nAvailable audio devices:")
    print("idx | in | out | default_sr | hostapi | name")
    print("----+----+-----+------------+---------+-----------------------------")
    for row in rows:
        print(
            f"{row['index']:>3} | {row['max_input_channels']:>2} | {row['max_output_channels']:>3} | "
            f"{row['default_samplerate']:>10} | {row['hostapi'][:18]:<18} | {row['name']}"
        )


def make_audio_bridge(enabled: bool, choice: AudioDeviceChoice, preamp_db: float) -> NullAudioBridge | RealtimeAudioBridge:
    if not enabled:
        return NullAudioBridge()
    try:
        return RealtimeAudioBridge(choice=choice, preamp_db=preamp_db)
    except ModuleNotFoundError as exc:
        missing = exc.name or str(exc)
        raise ModuleNotFoundError(
            f"Missing audio runtime dependency: {missing}. "
            "For source run: python -m pip install -r requirements.txt. "
            "For exe: rebuild package with build_cli_package.py --recreate-venv."
        ) from exc
