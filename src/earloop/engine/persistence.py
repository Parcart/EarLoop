from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any
from datetime import datetime, timezone
from uuid import uuid4

from .types import DomainState, EngineAudioConfig, EngineConfig, PerceptualParams, PipelineConfig, SavedProfile


def resolve_engine_state_path() -> Path:
    configured = os.environ.get("EARLOOP_ENGINE_STATE_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    repo_root = Path(__file__).resolve().parents[3]
    return repo_root / "data" / "engine" / "domain-state.json"


def resolve_engine_user_meta_path() -> Path:
    configured = os.environ.get("EARLOOP_ENGINE_USER_META_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_engine_state_path().with_name("user-meta.json")


def resolve_engine_event_log_path() -> Path:
    configured = os.environ.get("EARLOOP_ENGINE_EVENT_LOG_PATH")
    if configured:
        return Path(configured).expanduser().resolve()
    return resolve_engine_state_path().with_name("session-events.jsonl")


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


def load_or_create_user_identity(path: Path | None = None) -> dict[str, str]:
    meta_path = path or resolve_engine_user_meta_path()
    if meta_path.exists():
        try:
            raw = json.loads(meta_path.read_text(encoding="utf-8"))
            if isinstance(raw, dict) and isinstance(raw.get("userId"), str) and raw.get("userId"):
                created_at = str(raw.get("createdAt") or "")
                app_build_version = str(raw.get("appBuildVersion") or "")
                return {
                    "userId": raw["userId"],
                    "createdAt": created_at or datetime.now(timezone.utc).isoformat(),
                    "appBuildVersion": app_build_version,
                }
        except Exception:
            pass

    identity = {
        "userId": str(uuid4()),
        "createdAt": datetime.now(timezone.utc).isoformat(),
        "appBuildVersion": "",
    }
    save_user_identity(identity, meta_path)
    return identity


def save_user_identity(identity: dict[str, str], path: Path | None = None) -> Path:
    meta_path = path or resolve_engine_user_meta_path()
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "userId": str(identity.get("userId") or ""),
        "createdAt": str(identity.get("createdAt") or ""),
        "appBuildVersion": str(identity.get("appBuildVersion") or ""),
    }
    temp_path = meta_path.with_suffix(f"{meta_path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(meta_path)
    return meta_path


def append_event_log_entry(entry: dict[str, Any], path: Path | None = None) -> Path:
    log_path = path or resolve_engine_event_log_path()
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(entry, ensure_ascii=False))
        fp.write("\n")
    return log_path
