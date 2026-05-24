from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TextIO

from earloop.utils.logging_utils import setup_logger

from .audio_runtime import AudioRuntimeController
from .protocol import (
    EngineRequest,
    EngineResponse,
    ProtocolValidationError,
    build_error_response,
    build_success_response,
    parse_request,
)
from .types import EngineConfig, PerceptualParams, SavedProfile


HandlerFn = Callable[[dict[str, Any]], dict[str, Any] | list[Any] | None]


@dataclass(slots=True)
class AudioBridgeCommandRouter:
    audio_runtime: AudioRuntimeController
    logger = setup_logger("earloop.audio-bridge-router")
    started_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    _shutdown_requested: bool = False

    def dispatch(self, request: EngineRequest) -> EngineResponse:
        handler = self._handlers().get(request.command)
        if handler is None:
            return build_error_response(
                request.request_id,
                code="unsupported_command",
                message=f"Unsupported command: {request.command}",
            )
        try:
            return build_success_response(request.request_id, handler(request.payload))
        except KeyError as exc:
            self.logger.warning("request %s invalid payload: %s", request.request_id, exc.args[0])
            return build_error_response(
                request.request_id,
                code="invalid_payload",
                message=f"Missing payload field: {exc.args[0]}",
            )
        except Exception as exc:
            self.logger.exception("request %s command=%s failed", request.request_id, request.command)
            return build_error_response(
                request.request_id,
                code="handler_error",
                message=str(exc),
            )

    def _handlers(self) -> dict[str, HandlerFn]:
        return {
            "get_audio_bridge_health": self.handle_get_audio_bridge_health,
            "get_audio_status": self.handle_get_audio_status,
            "get_runtime_profile_status": self.handle_get_runtime_profile_status,
            "get_session_preview_status": self.handle_get_session_preview_status,
            "list_audio_devices": self.handle_list_audio_devices,
            "apply_engine_config": self.handle_apply_engine_config,
            "sync_runtime_profile": self.handle_sync_runtime_profile,
            "apply_session_preview": self.handle_apply_session_preview,
            "shutdown_audio_bridge": self.handle_shutdown_audio_bridge,
        }

    def handle_get_audio_bridge_health(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {
            "implementation": "python",
            "state": "running",
            "ready": True,
            "pid": os.getpid(),
            "startedAt": self.started_at,
            "lastError": None,
        }

    def handle_get_audio_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.get_status().to_dict()

    def handle_get_runtime_profile_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.get_runtime_profile_status().to_dict()

    def handle_get_session_preview_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.get_session_preview_status().to_dict()

    def handle_list_audio_devices(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.list_audio_devices().to_dict()

    def handle_apply_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = EngineConfig.from_dict(payload["config"])
        return self.audio_runtime.apply_engine_config(config).to_dict()

    def handle_sync_runtime_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = EngineConfig.from_dict(payload["config"])
        active_profile_payload = payload.get("activeProfile")
        active_profile = SavedProfile.from_dict(active_profile_payload) if active_profile_payload is not None else None
        return self.audio_runtime.set_processing_enabled(
            config.processing_enabled,
            active_profile,
            config,
        ).to_dict()

    def handle_apply_session_preview(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.audio_runtime.apply_session_preview(
            session_id=str(payload["sessionId"]),
            target=str(payload["target"]),
            params=PerceptualParams.from_dict(payload["params"]),
            label=str(payload["label"]),
            config=EngineConfig.from_dict(payload["config"]),
        ).to_dict()

    def handle_shutdown_audio_bridge(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._shutdown_requested = True
        self.audio_runtime.shutdown()
        self.logger.info("shutdown requested")
        return {
            "state": "stopping",
            "pid": os.getpid(),
        }


@dataclass(slots=True)
class AudioBridgeProtocolServer:
    router: AudioBridgeCommandRouter = field(default_factory=lambda: AudioBridgeCommandRouter(AudioRuntimeController()))
    logger = setup_logger("earloop.audio-bridge-server")

    def handle_message(self, raw: str | bytes | dict) -> EngineResponse:
        request_id = "unknown"
        try:
            request = parse_request(raw)
            request_id = request.request_id
            self.logger.info("request %s command=%s", request.request_id, request.command)
            return self.router.dispatch(request)
        except json.JSONDecodeError as exc:
            self.logger.warning("invalid json for request %s: %s", request_id, exc)
            return build_error_response(
                request_id,
                code="invalid_json",
                message="Request is not valid JSON",
                details={"error": str(exc)},
            )
        except ProtocolValidationError as exc:
            self.logger.warning("invalid request %s: %s", request_id, exc)
            return build_error_response(
                request_id,
                code="invalid_request",
                message=str(exc),
                details=exc.details,
            )

    def handle_json(self, raw: str | bytes | dict) -> str:
        return self.handle_message(raw).to_json()

    def serve_stdio(self, stdin: TextIO | None = None, stdout: TextIO | None = None) -> None:
        in_stream = stdin or sys.stdin
        out_stream = stdout or sys.stdout
        for line in in_stream:
            message = line.strip()
            if not message:
                continue
            response = self.handle_json(message)
            out_stream.write(response + "\n")
            out_stream.flush()
            if self.router._shutdown_requested:
                self.logger.info("audio bridge stdio server stopping on shutdown request")
                break


def run_audio_bridge_worker_stdio() -> None:
    setup_logger("earloop.audio-bridge-server").info("audio bridge stdio server starting")
    AudioBridgeProtocolServer().serve_stdio()


if __name__ == "__main__":
    run_audio_bridge_worker_stdio()
