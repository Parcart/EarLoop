from __future__ import annotations

import atexit
import json
import os
import subprocess
import sys
import threading
from datetime import datetime, timezone
from pathlib import Path
from queue import Empty, Queue
from typing import Any
from uuid import uuid4

from earloop.utils.logging_utils import setup_logger


class DomainWorkerClient:
    def __init__(self) -> None:
        self._logger = setup_logger("earloop.domain-worker-client")
        self._process: subprocess.Popen[str] | None = None
        self._stdout_thread: threading.Thread | None = None
        self._stderr_thread: threading.Thread | None = None
        self._wait_thread: threading.Thread | None = None
        self._pending: dict[str, Queue[Any]] = {}
        self._pending_lock = threading.Lock()
        self._write_lock = threading.Lock()
        self._status_lock = threading.Lock()
        self._start_lock = threading.Lock()
        self._expected_stop = False
        self._status: dict[str, Any] = {
            "implementation": "python",
            "state": "stopped",
            "ready": False,
            "pid": None,
            "lastError": None,
            "lastStartedAt": None,
            "lastStoppedAt": None,
            "lastResponseAt": None,
            "lastExitCode": None,
            "commandInFlightCount": 0,
        }
        atexit.register(self.stop)

    def get_status(self) -> dict[str, Any]:
        with self._status_lock:
            return dict(self._status)

    def _set_status(self, **updates: Any) -> None:
        with self._status_lock:
            self._status.update(updates)

    def _set_pending_count(self) -> None:
        with self._pending_lock:
            pending_count = len(self._pending)
        self._set_status(commandInFlightCount=pending_count)

    def _build_command(self) -> list[str]:
        if getattr(sys, "frozen", False):
            return [sys.executable]
        # Never execute src/earloop/engine/server.py directly: that puts the engine
        # package directory first on sys.path and shadows stdlib `types`.
        return [sys.executable, "-m", "earloop.engine.domain_worker"]

    def _build_env(self) -> dict[str, str]:
        env = dict(os.environ)
        env["EARLOOP_PROCESS_ROLE"] = "domain_worker"
        if not getattr(sys, "frozen", False):
            src_root = str(Path(__file__).resolve().parents[2])
            existing_pythonpath = env.get("PYTHONPATH")
            env["PYTHONPATH"] = src_root if not existing_pythonpath else os.pathsep.join([src_root, existing_pythonpath])
        return env

    def ensure_started(self) -> None:
        with self._start_lock:
            process = self._process
            if process is not None and process.poll() is None:
                return

            command = self._build_command()
            self._logger.info("starting domain worker process: command=%s", command)
            process = subprocess.Popen(
                command,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
                encoding="utf-8",
                bufsize=1,
                cwd=os.getcwd(),
                env=self._build_env(),
            )
            self._process = process
            self._expected_stop = False
            started_at = datetime.now(timezone.utc).isoformat()
            self._set_status(
                state="starting",
                ready=False,
                pid=process.pid,
                lastError=None,
                lastStartedAt=started_at,
                lastExitCode=None,
            )
            self._stdout_thread = threading.Thread(target=self._stdout_loop, args=(process,), name="domain-worker-stdout", daemon=True)
            self._stderr_thread = threading.Thread(target=self._stderr_loop, args=(process,), name="domain-worker-stderr", daemon=True)
            self._wait_thread = threading.Thread(target=self._wait_loop, args=(process,), name="domain-worker-wait", daemon=True)
            self._stdout_thread.start()
            self._stderr_thread.start()
            self._wait_thread.start()

    def request(self, command: str, payload: dict[str, Any] | None = None, *, timeout: float = 6.0) -> Any:
        self.ensure_started()
        process = self._process
        if process is None or process.stdin is None:
            raise RuntimeError("Domain worker stdin is unavailable")

        request_id = str(uuid4())
        mailbox: Queue[Any] = Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = mailbox
        self._set_pending_count()

        message = json.dumps(
            {
                "request_id": request_id,
                "command": command,
                "payload": payload or {},
            },
            ensure_ascii=True,
        )

        try:
            with self._write_lock:
                process.stdin.write(message + "\n")
                process.stdin.flush()
        except Exception:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._set_pending_count()
            self._set_status(state="failed", ready=False, lastError="Failed to write request to domain worker")
            raise

        try:
            response = mailbox.get(timeout=timeout)
        except Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._set_pending_count()
            timeout_error = f"Domain worker request timed out: {command}"
            self._set_status(state="degraded", ready=False, lastError=timeout_error)
            raise TimeoutError(timeout_error) from exc

        if isinstance(response, Exception):
            raise response
        if not response.get("ok"):
            error = response.get("error") or {}
            raise RuntimeError(str(error.get("message") or f"Domain worker command failed: {command}"))

        self._set_status(
            state="running",
            ready=True,
            lastError=None,
            lastResponseAt=datetime.now(timezone.utc).isoformat(),
        )
        return response.get("result")

    def stop(self) -> None:
        process = self._process
        if process is None:
            return
        self._logger.info("stopping domain worker process: pid=%s", process.pid)
        self._expected_stop = True
        stopped_at = datetime.now(timezone.utc).isoformat()
        try:
            self._request_existing_process("shutdown_domain_worker", {}, timeout=1.5)
        except Exception:
            self._logger.warning("graceful domain worker shutdown failed; killing process pid=%s", process.pid)
        try:
            process.wait(timeout=2.0)
        except Exception:
            try:
                process.kill()
            except Exception:
                pass
        self._set_status(
            state="stopped",
            ready=False,
            pid=None,
            lastError=None,
            lastStoppedAt=stopped_at,
        )

    def _request_existing_process(self, command: str, payload: dict[str, Any], *, timeout: float) -> Any:
        process = self._process
        if process is None or process.poll() is not None or process.stdin is None:
            raise RuntimeError("Domain worker is not running")

        request_id = str(uuid4())
        mailbox: Queue[Any] = Queue(maxsize=1)
        with self._pending_lock:
            self._pending[request_id] = mailbox
        self._set_pending_count()

        message = json.dumps(
            {
                "request_id": request_id,
                "command": command,
                "payload": payload,
            },
            ensure_ascii=True,
        )
        with self._write_lock:
            process.stdin.write(message + "\n")
            process.stdin.flush()

        try:
            response = mailbox.get(timeout=timeout)
        except Empty as exc:
            with self._pending_lock:
                self._pending.pop(request_id, None)
            self._set_pending_count()
            raise TimeoutError(f"Domain worker request timed out: {command}") from exc

        if isinstance(response, Exception):
            raise response
        return response.get("result")

    def _stdout_loop(self, process: subprocess.Popen[str]) -> None:
        if process is None or process.stdout is None:
            return
        for raw_line in process.stdout:
            line = raw_line.strip()
            if not line:
                continue
            try:
                response = json.loads(line)
            except Exception as exc:
                self._logger.warning("failed to parse domain worker stdout: %s | line=%s", exc, line)
                continue

            request_id = str(response.get("request_id") or "")
            with self._pending_lock:
                mailbox = self._pending.pop(request_id, None)
            self._set_pending_count()
            if mailbox is None:
                self._logger.warning("unmatched domain worker response: %s", response)
                continue
            mailbox.put(response)

    def _stderr_loop(self, process: subprocess.Popen[str]) -> None:
        if process is None or process.stderr is None:
            return
        for raw_line in process.stderr:
            line = raw_line.rstrip()
            if not line:
                continue
            self._logger.info("domain worker stderr: %s", line)

    def _wait_loop(self, process: subprocess.Popen[str]) -> None:
        exit_code = process.wait()
        stopped_at = datetime.now(timezone.utc).isoformat()
        expected_stop = self._expected_stop
        error = RuntimeError(f"Domain worker exited with code {exit_code}")
        if expected_stop:
            self._logger.info("domain worker stopped: pid=%s code=%s", process.pid, exit_code)
        else:
            self._logger.warning("domain worker exited unexpectedly: pid=%s code=%s", process.pid, exit_code)
        with self._pending_lock:
            pending = list(self._pending.values())
            self._pending.clear()
        self._set_pending_count()
        for mailbox in pending:
            mailbox.put(error)
        self._set_status(
            state="stopped",
            ready=False,
            pid=None,
            lastError=None if expected_stop else str(error),
            lastStoppedAt=stopped_at,
            lastExitCode=exit_code,
        )
        if self._process is process:
            self._process = None
