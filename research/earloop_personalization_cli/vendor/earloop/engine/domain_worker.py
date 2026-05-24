from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Callable, TextIO

from earloop.utils.logging_utils import setup_logger

from .protocol import (
    EngineRequest,
    EngineResponse,
    ProtocolValidationError,
    build_error_response,
    build_success_response,
    parse_request,
)
from .storage import InMemoryEngineStorage


HandlerFn = Callable[[dict[str, Any]], dict[str, Any] | list[Any] | None]


@dataclass(slots=True)
class DomainCommandRouter:
    storage: InMemoryEngineStorage
    logger = setup_logger("earloop.domain-worker-router")
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
            "get_main_state": self.handle_get_main_state,
            "get_engine_status": self.handle_get_engine_status,
            "list_profiles": self.handle_list_profiles,
            "get_domain_worker_health": self.handle_get_domain_worker_health,
            "resolve_session_preview_target": self.handle_resolve_session_preview_target,
            "set_active_profile": self.handle_set_active_profile,
            "set_processing_enabled": self.handle_set_processing_enabled,
            "save_profile": self.handle_save_profile,
            "save_profile_from_session": self.handle_save_profile_from_session,
            "update_profile": self.handle_update_profile,
            "delete_profile": self.handle_delete_profile,
            "get_engine_config": self.handle_get_engine_config,
            "update_engine_config": self.handle_update_engine_config,
            "get_domain_state": self.handle_get_domain_state,
            "create_session": self.handle_create_session,
            "start_session": self.handle_start_session,
            "generate_next_pair": self.handle_generate_next_pair,
            "shutdown_domain_worker": self.handle_shutdown_domain_worker,
        }

    def handle_get_main_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_main_state().to_dict()

    def handle_get_engine_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_engine_status().to_dict()

    def handle_list_profiles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        del payload
        return [profile.to_dict() for profile in self.storage.get_profiles()]

    def handle_get_domain_worker_health(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return {
            "implementation": "python",
            "state": "running",
            "ready": True,
            "pid": os.getpid(),
            "startedAt": self.started_at,
            "lastError": None,
        }

    def handle_resolve_session_preview_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        params, label = self.storage.resolve_session_preview_target(
            session_id=str(payload["sessionId"]),
            target=str(payload["target"]),
        )
        return {
            "params": params.to_dict(),
            "label": label,
        }

    def handle_set_active_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.set_active_profile(profile_id=str(payload["profileId"])).to_dict()

    def handle_set_processing_enabled(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.set_processing_enabled(enabled=bool(payload["enabled"])).to_dict()

    def handle_save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.save_profile(
            name=str(payload["name"]),
            final_choice=str(payload["finalChoice"]),
            pair_a=payload["pairA"],
            pair_b=payload["pairB"],
            pipeline_config=payload["pipelineConfig"],
            session_base_params=payload["sessionBaseParams"],
        )

    def handle_save_profile_from_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.save_profile_from_session(
            name=str(payload["name"]),
            final_choice=str(payload.get("finalChoice", "base")),
        )

    def handle_update_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            profile.to_dict()
            for profile in self.storage.update_profile(
                profile_id=str(payload["profileId"]),
                name=str(payload["name"]),
                params=payload["params"],
            )
        ]

    def handle_delete_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        return [
            profile.to_dict()
            for profile in self.storage.delete_profile(profile_id=str(payload["profileId"]))
        ]

    def handle_get_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_engine_config().to_dict()

    def handle_update_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.update_engine_config(payload["config"]).to_dict()

    def handle_get_domain_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_domain_state().to_dict()

    def handle_create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.create_session(payload).to_dict()

    def handle_start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.start_session(payload).to_dict()

    def handle_generate_next_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.generate_next_pair(payload).to_dict()

    def handle_shutdown_domain_worker(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._shutdown_requested = True
        self.logger.info("shutdown requested")
        return {
            "state": "stopping",
            "pid": os.getpid(),
        }


@dataclass(slots=True)
class DomainWorkerProtocolServer:
    router: DomainCommandRouter = field(default_factory=lambda: DomainCommandRouter(InMemoryEngineStorage()))
    logger = setup_logger("earloop.domain-worker-server")

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
                self.logger.info("domain worker stdio server stopping on shutdown request")
                break


def run_domain_worker_stdio() -> None:
    setup_logger("earloop.domain-worker-server").info("domain worker stdio server starting")
    DomainWorkerProtocolServer().serve_stdio()


if __name__ == "__main__":
    run_domain_worker_stdio()
