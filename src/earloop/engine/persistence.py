from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

from .types import DomainState, EngineAudioConfig, EngineConfig, PerceptualParams, PipelineConfig, SavedProfile


def resolve_engine_state_path() -> Path:
    configured = os.environ.get("EARLOOP_ENGINE_STATE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "engine" / "domain-state.json"


def _profile_from_dict(payload: dict[str, Any]) -> SavedProfile:
    return SavedProfile(
        profile_id=str(payload["id"]),
        name=str(payload["name"]),
        params=PerceptualParams.from_dict(payload["params"]),
        pipeline_config=PipelineConfig.from_dict(payload["pipelineConfig"]),
    )


def _config_from_dict(payload: dict[str, Any]) -> EngineConfig:
    audio = payload["audio"]
    defaults = payload["defaults"]
    runtime = payload.get("runtime", {})
    return EngineConfig(
        audio=EngineAudioConfig(
            input_device_id=str(audio["inputDeviceId"]),
            output_device_id=str(audio["outputDeviceId"]),
            sample_rate=str(audio["sampleRate"]),
            channels=str(audio["channels"]),
        ),
        active_profile_id=str(defaults["activeProfileId"]),
        processing_enabled=bool(runtime.get("processingEnabled", True)),
    )


def load_persisted_domain_state(path: Path | None = None) -> DomainState | None:
    state_path = path or resolve_engine_state_path()
    if not state_path.exists():
        return None

    raw = json.loads(state_path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError("Persisted engine state must be a JSON object")

    profiles_payload = raw.get("profiles", [])
    config_payload = raw.get("config")
    if not isinstance(profiles_payload, list) or not isinstance(config_payload, dict):
        raise ValueError("Persisted engine state is missing profiles/config")

    profiles = [_profile_from_dict(item) for item in profiles_payload]
    config = _config_from_dict(config_payload)
    return DomainState(
        profiles=profiles,
        config=config,
        session=None,
    )


def save_persisted_domain_state(state: DomainState, path: Path | None = None) -> Path:
    state_path = path or resolve_engine_state_path()
    state_path.parent.mkdir(parents=True, exist_ok=True)

    payload = {
        "profiles": [profile.to_dict() for profile in state.profiles],
        "config": state.config.to_dict(),
    }

    temp_path = state_path.with_suffix(f"{state_path.suffix}.tmp")
    temp_path.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    temp_path.replace(state_path)
    return state_path
