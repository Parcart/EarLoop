from __future__ import annotations

import json
import os
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any


def _env_bool(name: str, default: bool) -> bool:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in {"1", "true", "yes", "y", "on"}


def _env_int(name: str, default: int | None) -> int | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_list(name: str, default: list[str] | None) -> list[str] | None:
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    value = raw.strip()
    if value.lower() in {"all", "any", "none", "*", "0"}:
        return []
    return [item.strip() for item in value.split(",") if item.strip()]


@dataclass
class AudioConfig:
    enabled: bool = True
    capture_device: int | None = None
    playback_device: int | None = None
    auto_detect_devices: bool = True
    latency: str = "low"
    confirm_devices: bool = True
    # For Windows tester builds MME is usually the most stable PortAudio host API.
    # Set to [] or null in config.json to show/use all host APIs.
    allowed_host_apis: list[str] = field(default_factory=lambda: ["MME"])
    # Show warning when VB-Cable endpoints are not available. Testers can still continue
    # with an analog/physical capture device, but the common VB-Cable setup should be obvious.
    warn_if_vb_cable_missing: bool = True
    # Check Windows/PortAudio current default output before session start.
    # auto: if capture is CABLE Output, expect system output = CABLE Input.
    # vb_cable: always expect system output = CABLE Input.
    # selected_playback: expect system output = selected playback device.
    # none/off: do not validate default output.
    expected_system_output: str = "auto"
    # On some Windows/MME + VB-Cable setups the bridge must be started while
    # Windows Output is still the real headphones/speakers, then the tester
    # switches Windows Output to CABLE Input after [audio] bridge started.
    pre_bridge_headphones_prompt: bool = True
    post_bridge_vb_switch_prompt: bool = True
    # Desktop-like lifecycle: keep one bridge alive while several sessions run.
    # This avoids restarting PortAudio/MME streams while Windows Output is already CABLE Input.
    keep_bridge_alive_between_sessions: bool = True


@dataclass
class MapperConfig:
    # auto = use TorchScript model_path when it exists, otherwise safe interpretable fallback.
    # torchscript = require TorchScript unless allow_interpretable_fallback=true.
    # interpretable = manual deterministic contract mapper.
    mode: str = "auto"
    model_path: str = "models/model_b_contract_controllable_mlp.npz"
    device: str = "cpu"
    allow_interpretable_fallback: bool = True


@dataclass
class LoggingConfig:
    save_step_vectors: bool = True
    save_eq_curves: bool = True
    save_model_state_each_step: bool = True
    zip_on_finish: bool = True


@dataclass
class AppConfig:
    app_name: str = "EarLoop Personalization CLI"
    default_strategy: str = "phase_aware_feedback"
    budget_steps: int = 24
    sample_rate: int = 48000
    channels: int = 2
    blocksize: int = 1024
    preamp_db: float = 0.0
    max_abs_db: float = 12.0
    data_dir: str = "runtime_data"
    audio: AudioConfig = field(default_factory=AudioConfig)
    mapper: MapperConfig = field(default_factory=MapperConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)

    @property
    def data_path(self) -> Path:
        return Path(self.data_dir).resolve()


def _merge_dict(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in override.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = _merge_dict(out[k], v)
        else:
            out[k] = v
    return out


def _apply_env_overrides(payload: dict[str, Any]) -> dict[str, Any]:
    # Useful in PyCharm Run Configuration and tester shortcuts.
    if os.environ.get("EARLOOP_CLI_DATA_DIR"):
        payload["data_dir"] = os.environ["EARLOOP_CLI_DATA_DIR"]
    if os.environ.get("EARLOOP_CLI_STRATEGY"):
        payload["default_strategy"] = os.environ["EARLOOP_CLI_STRATEGY"]
    if os.environ.get("EARLOOP_CLI_BUDGET"):
        try:
            payload["budget_steps"] = int(os.environ["EARLOOP_CLI_BUDGET"])
        except ValueError:
            pass

    audio = dict(payload.get("audio", {}))
    audio["enabled"] = _env_bool("EARLOOP_CLI_AUDIO_ENABLED", bool(audio.get("enabled", True)))
    audio["capture_device"] = _env_int("EARLOOP_CLI_CAPTURE_DEVICE", audio.get("capture_device"))
    audio["playback_device"] = _env_int("EARLOOP_CLI_PLAYBACK_DEVICE", audio.get("playback_device"))
    audio["auto_detect_devices"] = _env_bool("EARLOOP_CLI_AUTO_DETECT_DEVICES", bool(audio.get("auto_detect_devices", True)))
    audio["confirm_devices"] = _env_bool("EARLOOP_CLI_CONFIRM_DEVICES", bool(audio.get("confirm_devices", True)))
    audio["allowed_host_apis"] = _env_list("EARLOOP_CLI_ALLOWED_HOST_APIS", audio.get("allowed_host_apis", ["MME"]))
    audio["warn_if_vb_cable_missing"] = _env_bool("EARLOOP_CLI_WARN_IF_VB_CABLE_MISSING", bool(audio.get("warn_if_vb_cable_missing", True)))
    if os.environ.get("EARLOOP_CLI_EXPECTED_SYSTEM_OUTPUT"):
        audio["expected_system_output"] = os.environ["EARLOOP_CLI_EXPECTED_SYSTEM_OUTPUT"]
    audio["pre_bridge_headphones_prompt"] = _env_bool("EARLOOP_CLI_PRE_BRIDGE_HEADPHONES_PROMPT", bool(audio.get("pre_bridge_headphones_prompt", True)))
    audio["post_bridge_vb_switch_prompt"] = _env_bool("EARLOOP_CLI_POST_BRIDGE_VB_SWITCH_PROMPT", bool(audio.get("post_bridge_vb_switch_prompt", True)))
    audio["keep_bridge_alive_between_sessions"] = _env_bool("EARLOOP_CLI_KEEP_BRIDGE_ALIVE", bool(audio.get("keep_bridge_alive_between_sessions", True)))
    payload["audio"] = audio

    mapper = dict(payload.get("mapper", {}))
    if os.environ.get("EARLOOP_CLI_MAPPER_MODE"):
        mapper["mode"] = os.environ["EARLOOP_CLI_MAPPER_MODE"]
    if os.environ.get("EARLOOP_CLI_MAPPER_PATH"):
        mapper["model_path"] = os.environ["EARLOOP_CLI_MAPPER_PATH"]
    if os.environ.get("EARLOOP_CLI_MAPPER_DEVICE"):
        mapper["device"] = os.environ["EARLOOP_CLI_MAPPER_DEVICE"]
    payload["mapper"] = mapper
    return payload


def load_config(path: str | Path | None = None) -> AppConfig:
    if path is None:
        path = os.environ.get("EARLOOP_CLI_CONFIG", "config.json")
    default = asdict(AppConfig())
    if path is not None and Path(path).exists():
        user = json.loads(Path(path).read_text(encoding="utf-8"))
        default = _merge_dict(default, user)
    default = _apply_env_overrides(default)
    audio = AudioConfig(**default.get("audio", {}))
    mapper = MapperConfig(**default.get("mapper", {}))
    logging = LoggingConfig(**default.get("logging", {}))
    payload = {k: v for k, v in default.items() if k not in {"audio", "mapper", "logging"}}
    return AppConfig(**payload, audio=audio, mapper=mapper, logging=logging)


def save_config(config: AppConfig, path: str | Path = "config.json") -> None:
    Path(path).write_text(json.dumps(asdict(config), ensure_ascii=False, indent=2), encoding="utf-8")


def persist_audio_selection(
    path: str | Path,
    *,
    capture_device: int | None,
    playback_device: int | None,
    allowed_host_apis: list[str] | None = None,
) -> None:
    """Persist the tester's chosen devices in config.json without touching logs/profiles.

    This prevents the next run from auto-detecting the wrong VB-Cable endpoint.
    """
    config_path = Path(path)
    if config_path.exists():
        try:
            payload = json.loads(config_path.read_text(encoding="utf-8"))
        except Exception:
            payload = asdict(AppConfig())
    else:
        payload = asdict(AppConfig())
    audio = dict(payload.get("audio", {}))
    audio["capture_device"] = capture_device
    audio["playback_device"] = playback_device
    audio["auto_detect_devices"] = False
    audio["confirm_devices"] = True
    if allowed_host_apis is not None:
        audio["allowed_host_apis"] = allowed_host_apis
    payload["audio"] = audio
    config_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def save_config_template(path: str | Path) -> None:
    Path(path).write_text(json.dumps(asdict(AppConfig()), ensure_ascii=False, indent=2), encoding="utf-8")
