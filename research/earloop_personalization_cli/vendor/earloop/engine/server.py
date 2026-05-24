from __future__ import annotations

import json
import os
import sys
from dataclasses import dataclass, field
from typing import TextIO

from earloop.utils.logging_utils import setup_logger

from .audio_runtime import AudioRuntimeController
from .domain_client import DomainWorkerClient
from .domain_worker import run_domain_worker_stdio
from .handlers import EngineCommandRouter
from .persistence import describe_runtime_paths
from .protocol import (
    EngineResponse,
    ProtocolValidationError,
    build_error_response,
    parse_request,
)


@dataclass(slots=True)
class EngineProtocolServer:
    router: EngineCommandRouter = field(default_factory=lambda: EngineCommandRouter(
        DomainWorkerClient(),
        AudioRuntimeController(),
    ))
    logger = setup_logger("earloop.engine-server")

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


def run_stdio_server() -> None:
    setup_logger("earloop.engine-server").info(
        "engine stdio server starting with runtime paths: %s",
        describe_runtime_paths(),
    )
    EngineProtocolServer().serve_stdio()


_DEBUGGER_ATTACHED = False

def _try_attach_debugger() -> None:
    global _DEBUGGER_ATTACHED
    if _DEBUGGER_ATTACHED:
        return

    if os.getenv("EARLOOP_PYCHARM_DEBUG") != "1":
        return

    host = os.getenv("EARLOOP_PYCHARM_DEBUG_HOST", "127.0.0.1")
    port = int(os.getenv("EARLOOP_PYCHARM_DEBUG_PORT", "5678"))
    suspend = os.getenv("EARLOOP_PYCHARM_DEBUG_SUSPEND", "0") == "1"

    try:
        import pydevd_pycharm
        pydevd_pycharm.settrace(
            host,
            port=port,
            stdout_to_server=True,
            stderr_to_server=True,
            suspend=suspend,
        )
        _DEBUGGER_ATTACHED = True
        print(f"[engine] debugger attached to {host}:{port}", file=sys.stderr, flush=True)
    except Exception as exc:
        print(f"[engine] failed to attach debugger: {exc}", file=sys.stderr, flush=True)


def main() -> None:
    _try_attach_debugger()
    if os.getenv("EARLOOP_PROCESS_ROLE") == "domain_worker":
        run_domain_worker_stdio()
    else:
        run_stdio_server()


if __name__ == "__main__":
    main()
