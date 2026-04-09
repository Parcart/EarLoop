from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

from earloop.utils.logging_utils import setup_logger

from .audio_runtime import AudioRuntimeController
from .protocol import EngineRequest, build_error_response, build_success_response
from .storage import InMemoryEngineStorage


HandlerFn = Callable[[dict[str, Any]], dict[str, Any] | list[Any] | None]


@dataclass(slots=True)
class EngineCommandRouter:
    storage: InMemoryEngineStorage
    audio_runtime: AudioRuntimeController
    logger = setup_logger("earloop.engine-router")

    def _resolve_audio_device_labels(self, input_device_id: str | None, output_device_id: str | None) -> dict[str, str | None]:
        try:
            catalog = self.audio_runtime.list_audio_devices()
            devices_by_id = {
                device.device_id: device.label
                for device in [*catalog.inputs, *catalog.outputs]
            }
        except Exception:
            devices_by_id = {}
        return {
            "input": devices_by_id.get(input_device_id or "", input_device_id),
            "output": devices_by_id.get(output_device_id or "", output_device_id),
        }

    def _log_update_engine_config_result(self, status) -> None:
        attempted_config = status.desired_config.to_dict() if status.desired_config is not None else None
        fallback_active_config = status.active_config.to_dict() if status.active_config is not None else None
        attempted_labels = self._resolve_audio_device_labels(
            status.desired_config.input_device_id if status.desired_config is not None else None,
            status.desired_config.output_device_id if status.desired_config is not None else None,
        )
        fallback_active_labels = self._resolve_audio_device_labels(
            status.active_config.input_device_id if status.active_config is not None else None,
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
        config = self.storage.get_engine_config()
        active_profile = self.storage.get_active_profile()
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
        state = self.storage.get_main_state()
        state.audio_status = self.audio_runtime.get_status()
        state.runtime_profile_status = self.audio_runtime.get_runtime_profile_status()
        return state.to_dict()

    def handle_get_engine_status(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        self._sync_runtime_profile()
        status = self.storage.get_engine_status()
        status.audio_status = self.audio_runtime.get_status()
        status.runtime_profile_status = self.audio_runtime.get_runtime_profile_status()
        return status.to_dict()

    def handle_list_audio_devices(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.audio_runtime.list_audio_devices().to_dict()

    def handle_list_profiles(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        del payload
        return [profile.to_dict() for profile in self.storage.get_profiles()]

    def handle_preview_session_target(self, payload: dict[str, Any]) -> dict[str, Any]:
        session_id = str(payload["sessionId"])
        target = str(payload["target"])
        self.logger.info("preview_session_target requested: session_id=%s target=%s", session_id, target)
        params, label = self.storage.resolve_session_preview_target(session_id=session_id, target=target)
        result = self.audio_runtime.apply_session_preview(
            session_id=session_id,
            target=target,
            params=params,
            label=label,
            config=self.storage.get_engine_config(),
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
        state = self.storage.set_active_profile(profile_id=str(payload["profileId"]))
        self._sync_runtime_profile()
        state.audio_status = self.audio_runtime.get_status()
        state.runtime_profile_status = self.audio_runtime.get_runtime_profile_status()
        return state.to_dict()

    def handle_set_processing_enabled(self, payload: dict[str, Any]) -> dict[str, Any]:
        self.logger.info("set_processing_enabled requested: enabled=%s", payload["enabled"])
        state = self.storage.set_processing_enabled(enabled=bool(payload["enabled"]))
        self._sync_runtime_profile()
        state.audio_status = self.audio_runtime.get_status()
        state.runtime_profile_status = self.audio_runtime.get_runtime_profile_status()
        return state.to_dict()

    def handle_save_profile(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.storage.save_profile(
            name=str(payload["name"]),
            final_choice=str(payload["finalChoice"]),
            pair_a=payload["pairA"],
            pair_b=payload["pairB"],
            pipeline_config=payload["pipelineConfig"],
            session_base_params=payload["sessionBaseParams"],
        )
        self._sync_runtime_profile()
        return result

    def handle_save_profile_from_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        result = self.storage.save_profile_from_session(
            name=str(payload["name"]),
            final_choice=str(payload.get("finalChoice", "base")),
        )
        self._sync_runtime_profile()
        return result

    def handle_update_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = [
            profile.to_dict()
            for profile in self.storage.update_profile(
                profile_id=str(payload["profileId"]),
                name=str(payload["name"]),
                params=payload["params"],
            )
        ]
        self._sync_runtime_profile()
        return result

    def handle_delete_profile(self, payload: dict[str, Any]) -> list[dict[str, Any]]:
        result = [
            profile.to_dict()
            for profile in self.storage.delete_profile(profile_id=str(payload["profileId"]))
        ]
        self._sync_runtime_profile()
        return result

    def handle_get_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_engine_config().to_dict()

    def handle_update_engine_config(self, payload: dict[str, Any]) -> dict[str, Any]:
        config = self.storage.update_engine_config(payload["config"])
        status = self.audio_runtime.apply_engine_config(config)
        self._sync_runtime_profile()
        self._log_update_engine_config_result(status)
        return config.to_dict()

    def handle_get_domain_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        del payload
        return self.storage.get_domain_state().to_dict()

    def handle_create_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.create_session(payload).to_dict()

    def handle_start_session(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.start_session(payload).to_dict()

    def handle_generate_next_pair(self, payload: dict[str, Any]) -> dict[str, Any]:
        return self.storage.generate_next_pair(payload).to_dict()
