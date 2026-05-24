from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from earloop.utils.logging_utils import setup_logger

from .audio_runtime import AudioRuntimeController
from .domain_client import DomainWorkerClient
from .protocol import EngineRequest, build_error_response, build_success_response
from .types import EngineConfig, PerceptualParams, SavedProfile


HandlerFn = Callable[[dict[str, Any]], dict[str, Any] | list[Any] | None]


@dataclass(slots=True)
class EngineCommandRouter:
    domain_client: DomainWorkerClient
    audio_runtime: AudioRuntimeController
    logger = setup_logger("earloop.engine-router")

    def _resolve_audio_device_labels(self, capture_device_id: str | None, output_device_id: str | None) -> dict[str, str | None]:
        try:
            catalog = self.audio_runtime.list_audio_devices()
            devices_by_id = {
                device.device_id: device.label
                for device in [*catalog.inputs, *catalog.outputs]
            }
        except Exception:
            devices_by_id = {}
        return {
            "capture": devices_by_id.get(capture_device_id or "", capture_device_id),
            "output": devices_by_id.get(output_device_id or "", output_device_id),
        }

    def _log_update_engine_config_result(self, status) -> None:
        attempted_config = status.desired_config.to_dict() if status.desired_config is not None else None
        fallback_active_config = status.active_config.to_dict() if status.active_config is not None else None
        attempted_labels = self._resolve_audio_device_labels(
            status.desired_config.capture_device_id if status.desired_config is not None else None,
            status.desired_config.output_device_id if status.desired_config is not None else None,
        )
        fallback_active_labels = self._resolve_audio_device_labels(
            status.active_config.capture_device_id if status.active_config is not None else None,
            status.active_config.output_device_id if status.active_config is not None else None,
        )

        if status.status == "applied":
            self.logger.info(
                "update_engine_config applied: status=%s active_config=%s active_device_labels=%s",
                status.status,
                fallback_active_config,
                fallback_active_labels,
            )
            return

        if status.active_config is not None:
            self.logger.warning(
                "update_engine_config failed; keeping previous active route | failure_status=%s details=%s attempted_config=%s attempted_devices=%s fallback_active_config=%s fallback_active_devices=%s kept_previous_active_route=%s",
                status.status,
                status.last_error,
                attempted_config,
                attempted_labels,
                fallback_active_config,
                fallback_active_labels,
                True,
            )
            return

        self.logger.error(
            "update_engine_config failed with no active route | failure_status=%s details=%s attempted_config=%s attempted_devices=%s fallback_active_config=%s fallback_active_devices=%s kept_previous_active_route=%s",
            status.status,
            status.last_error,
            attempted_config,
            attempted_labels,
            fallback_active_config,
            fallback_active_labels,
            False,
        )

    def _sync_runtime_profile(self) -> None:
        config = EngineConfig.from_dict(self.domain_client.request("get_engine_config", {}))
        profiles_payload = self.domain_client.request("list_profiles", {})
        active_profile = next(
            (
                SavedProfile.from_dict(profile)
                for profile in profiles_payload
                if str(profile.get("id")) == config.active_profile_id
            ),
            None,
        )
        self.audio_runtime.set_processing_enabled(config.processing_enabled, active_profile, config)

    def dispatch(self, request: EngineRequest):
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
            "list_audio_devices": self.handle_list_audio_devices,
            "list_profiles": self.handle_list_profiles,
            "preview_session_target": self.handle_preview_session_target,
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
        }

    def handle_get_main_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._sync_runtime_profile()
        state = dict(self.domain_client.request("get_main_state", {}))
        state["audioStatus"] = self.audio_runtime.get_status().to_dict()
        state["runtimeProfileStatus"] = self.audio_runtime.get_runtime_profile_status().to_dict()
        state["domainWorkerStatus"] = self.domain_client.get_status()
        return state

    def handle_get_engine_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._sync_runtime_profile()
        status = dict(self.domain_client.request("get_engine_status", {}))
        status["audioStatus"] = self.audio_runtime.get_status().to_dict()
        status["runtimeProfileStatus"] = self.audio_runtime.get_runtime_profile_status().to_dict()
        status["domainWorkerStatus"] = self.domain_client.get_status()
        return status

    def handle_list_audio_devices(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.list_audio_devices().to_dict()

    def handle_list_profiles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        del payload
        return self.domain_client.request("list_profiles", {})

    def handle_preview_session_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["sessionId"])
        target = str(payload["target"])
        self.logger.info("preview_session_target requested: session_id=%s target=%s", session_id, target)
        preview_target = self.domain_client.request(
            "resolve_session_preview_target",
            {
                "sessionId": session_id,
                "target": target,
            },
        )
        params = PerceptualParams.from_dict(preview_target["params"])
        label = str(preview_target["label"])
        result = self.audio_runtime.apply_session_preview(
            session_id=session_id,
            target=target,
            params=params,
            label=label,
            config=EngineConfig.from_dict(self.domain_client.request("get_engine_config", {})),
        ).to_dict()
        self.logger.info(
            "preview_session_target result: session_id=%s target=%s apply_status=%s applied_to_audio=%s last_error=%s",
            session_id,
            target,
            result.get("applyStatus"),
            result.get("appliedToAudio"),
            result.get("lastError"),
        )
        return result

    def handle_set_active_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.logger.info("set_active_profile requested: profile_id=%s", payload["profileId"])
        state = dict(self.domain_client.request("set_active_profile", payload))
        self._sync_runtime_profile()
        state["audioStatus"] = self.audio_runtime.get_status().to_dict()
        state["runtimeProfileStatus"] = self.audio_runtime.get_runtime_profile_status().to_dict()
        state["domainWorkerStatus"] = self.domain_client.get_status()
        return state

    def handle_set_processing_enabled(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.logger.info("set_processing_enabled requested: enabled=%s", payload["enabled"])
        state = dict(self.domain_client.request("set_processing_enabled", payload))
        self._sync_runtime_profile()
        state["audioStatus"] = self.audio_runtime.get_status().to_dict()
        state["runtimeProfileStatus"] = self.audio_runtime.get_runtime_profile_status().to_dict()
        state["domainWorkerStatus"] = self.domain_client.get_status()
        return state

    def handle_save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.domain_client.request("save_profile", payload)
        self._sync_runtime_profile()
        return result

    def handle_save_profile_from_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.domain_client.request("save_profile_from_session", payload)
        self._sync_runtime_profile()
        return result

    def handle_update_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = self.domain_client.request("update_profile", payload)
        self._sync_runtime_profile()
        return result

    def handle_delete_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = self.domain_client.request("delete_profile", payload)
        self._sync_runtime_profile()
        return result

    def handle_get_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.domain_client.request("get_engine_config", {})

    def handle_update_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = EngineConfig.from_dict(self.domain_client.request("update_engine_config", payload))
        status = self.audio_runtime.apply_engine_config(config)
        self._sync_runtime_profile()
        self._log_update_engine_config_result(status)
        return config.to_dict()

    def handle_get_domain_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.domain_client.request("get_domain_state", {})

    def handle_create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.domain_client.request("create_session", payload)

    def handle_start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.domain_client.request("start_session", payload)

    def handle_generate_next_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.domain_client.request("generate_next_pair", payload)
