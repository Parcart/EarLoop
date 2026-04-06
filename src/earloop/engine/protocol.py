from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from .types import CommandName, ProtocolError


COMMAND_ALIASES: dict[str, CommandName] = {
    "get_engine_status": "get_engine_status",
    "get_main_state": "get_main_state",
    "list_audio_devices": "list_audio_devices",
    "list_profiles": "list_profiles",
    "preview_session_target": "preview_session_target",
    "set_active_profile": "set_active_profile",
    "set_processing_enabled": "set_processing_enabled",
    "save_profile": "save_profile",
    "save_profile_from_session": "save_profile_from_session",
    "update_profile": "update_profile",
    "delete_profile": "delete_profile",
    "get_engine_config": "get_engine_config",
    "update_engine_config": "update_engine_config",
    "get_domain_state": "get_domain_state",
    "create_session": "create_session",
    "start_session": "start_session",
    "generate_next_pair": "generate_next_pair",
    "getEngineStatus": "get_engine_status",
    "getMainState": "get_main_state",
    "listAudioDevices": "list_audio_devices",
    "listProfiles": "list_profiles",
    "previewSessionTarget": "preview_session_target",
    "setActiveProfile": "set_active_profile",
    "setProcessingEnabled": "set_processing_enabled",
    "saveProfile": "save_profile",
    "saveProfileFromSession": "save_profile_from_session",
    "updateProfile": "update_profile",
    "deleteProfile": "delete_profile",
    "getEngineConfig": "get_engine_config",
    "updateEngineConfig": "update_engine_config",
    "getDomainState": "get_domain_state",
    "createSession": "create_session",
    "startSession": "start_session",
    "generateNextPair": "generate_next_pair",
}


class ProtocolValidationError(ValueError):
    def __init__(self, message: str, *, details: dict[str, Any] | None = None) -> None:
        super().__init__(message)
        self.details = details or {}


@dataclass(slots=True)
class EngineRequest:
    request_id: str
    command: CommandName
    payload: dict[str, Any]

    def to_dict(self) -> dict[str, Any]:
        return {
            "request_id": self.request_id,
            "command": self.command,
            "payload": self.payload,
        }

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)


@dataclass(slots=True)
class EngineResponse:
    request_id: str
    ok: bool
    result: dict[str, Any] | list[Any] | None = None
    error: ProtocolError | None = None

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "request_id": self.request_id,
            "ok": self.ok,
        }
        if self.ok:
            data["result"] = self.result
        else:
            data["error"] = self.error.to_dict() if self.error is not None else {
                "code": "unknown_error",
                "message": "Unknown protocol error",
            }
        return data

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), ensure_ascii=True)


def parse_request(raw: str | bytes | dict[str, Any]) -> EngineRequest:
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        payload = json.loads(raw)
    else:
        payload = raw

    if not isinstance(payload, dict):
        raise ProtocolValidationError("Request must be a JSON object")

    request_id = payload.get("request_id", payload.get("requestId"))
    command = payload.get("command")
    command_name = COMMAND_ALIASES.get(str(command)) if command is not None else None
    body = payload.get("payload", {})

    if not isinstance(request_id, str) or not request_id:
        raise ProtocolValidationError("request_id is required")
    if command_name is None:
        raise ProtocolValidationError("Unsupported command", details={"command": command})
    if body is None:
        body = {}
    if not isinstance(body, dict):
        raise ProtocolValidationError("payload must be an object")

    return EngineRequest(
        request_id=request_id,
        command=command_name,
        payload=body,
    )


def build_success_response(request_id: str, result: dict[str, Any] | list[Any] | None) -> EngineResponse:
    return EngineResponse(
        request_id=request_id,
        ok=True,
        result=result,
    )


def build_error_response(
    request_id: str,
    *,
    code: str,
    message: str,
    details: dict[str, Any] | None = None,
) -> EngineResponse:
    return EngineResponse(
        request_id=request_id,
        ok=False,
        error=ProtocolError(code=code, message=message, details=details or {}),
    )
