from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from typing import TYPE_CHECKING, Any, Literal

from earloop.utils.logging_utils import setup_logger

from .types import AudioDeviceInfo, AudioDevicesSnapshot, AudioStatusSnapshot, EngineAudioConfig, EngineConfig, PerceptualParams, RuntimeProfileStatusSnapshot, SavedProfile, SessionPreviewStatusSnapshot

if TYPE_CHECKING:
    from earloop.audio.engine import AudioEngine


AudioApplyState = Literal["idle", "applied", "failed", "pending_restart", "device_unavailable"]
EQ_BAND_FREQUENCIES = (25.0, 40.0, 63.0, 100.0, 160.0, 250.0, 400.0, 630.0, 1000.0, 2500.0, 6300.0, 16000.0)
WASAPI_HOSTAPI_NAME = "Windows WASAPI"


def _clone_audio_config(config: EngineAudioConfig | None) -> EngineAudioConfig | None:
    if config is None:
        return None
    return replace(config)


class AudioRuntimeController:
    def __init__(self) -> None:
        self._logger = setup_logger("earloop.audio-runtime")
        self._engine: AudioEngine | None = None
        self._pending_eq_profile: Any | None = None
        self._processing_enabled = True
        self._status = AudioStatusSnapshot(
            status="idle",
            active_config=None,
            desired_config=None,
            last_error=None,
            last_applied_at=None,
        )
        self._runtime_profile_status = RuntimeProfileStatusSnapshot(
            active_profile_id=None,
            active_profile_name=None,
            processor_mode="passthrough",
            eq_curve_ready=False,
            eq_band_count=None,
            preamp_db=None,
            applied_to_audio=False,
            apply_status="not_applied",
            last_error=None,
        )
        self._session_preview_status = SessionPreviewStatusSnapshot(
            session_id=None,
            target=None,
            label=None,
            processor_mode="passthrough",
            eq_curve_ready=False,
            eq_band_count=None,
            preamp_db=None,
            applied_to_audio=False,
            apply_status="idle",
            last_error=None,
        )

    def get_status(self) -> AudioStatusSnapshot:
        return AudioStatusSnapshot(
            status=self._status.status,
            active_config=_clone_audio_config(self._status.active_config),
            desired_config=_clone_audio_config(self._status.desired_config),
            last_error=self._status.last_error,
            last_applied_at=self._status.last_applied_at,
        )

    def get_runtime_profile_status(self) -> RuntimeProfileStatusSnapshot:
        return RuntimeProfileStatusSnapshot(
            active_profile_id=self._runtime_profile_status.active_profile_id,
            active_profile_name=self._runtime_profile_status.active_profile_name,
            processor_mode=self._runtime_profile_status.processor_mode,
            eq_curve_ready=self._runtime_profile_status.eq_curve_ready,
            eq_band_count=self._runtime_profile_status.eq_band_count,
            preamp_db=self._runtime_profile_status.preamp_db,
            applied_to_audio=self._runtime_profile_status.applied_to_audio,
            apply_status=self._runtime_profile_status.apply_status,
            last_error=self._runtime_profile_status.last_error,
        )

    def get_session_preview_status(self) -> SessionPreviewStatusSnapshot:
        return SessionPreviewStatusSnapshot(
            session_id=self._session_preview_status.session_id,
            target=self._session_preview_status.target,
            label=self._session_preview_status.label,
            processor_mode=self._session_preview_status.processor_mode,
            eq_curve_ready=self._session_preview_status.eq_curve_ready,
            eq_band_count=self._session_preview_status.eq_band_count,
            preamp_db=self._session_preview_status.preamp_db,
            applied_to_audio=self._session_preview_status.applied_to_audio,
            apply_status=self._session_preview_status.apply_status,
            last_error=self._session_preview_status.last_error,
        )

    def list_audio_devices(self) -> AudioDevicesSnapshot:
        audio_engine_cls, _, import_error = self._load_audio_runtime()
        if import_error is not None or audio_engine_cls is None:
            self._logger.warning("audio device enumeration unavailable: %s", import_error or "Audio backend is unavailable")
            return AudioDevicesSnapshot(inputs=[], outputs=[], status="unavailable", error=import_error or "Audio backend is unavailable")

        try:
            import sounddevice as sd
            devices, hostapis = self._load_device_catalog(audio_engine_cls)
            default_input, default_output = sd.default.device
        except Exception as exc:
            self._logger.exception("audio device enumeration failed")
            return AudioDevicesSnapshot(inputs=[], outputs=[], status="failed", error=str(exc))

        inputs: list[AudioDeviceInfo] = []
        outputs: list[AudioDeviceInfo] = []
        input_indexes_by_id: dict[str, int] = {}
        output_indexes_by_id: dict[str, int] = {}
        for index, device in enumerate(devices):
            hostapi_name = self._resolve_hostapi_name(device=device, hostapis=hostapis)
            device_id = str(index)
            label = str(device.get("name", device_id))
            default_sample_rate = self._normalize_sample_rate(device.get("default_samplerate"))
            if int(device.get("max_input_channels", 0)) > 0:
                inputs.append(AudioDeviceInfo(
                    device_id=device_id,
                    label=label,
                    kind="input",
                    is_default=index == default_input,
                    hostapi=hostapi_name,
                    max_input_channels=int(device.get("max_input_channels", 0)),
                    max_output_channels=int(device.get("max_output_channels", 0)),
                    default_sample_rate=default_sample_rate,
                ))
                input_indexes_by_id[device_id] = index
            if int(device.get("max_output_channels", 0)) > 0:
                outputs.append(AudioDeviceInfo(
                    device_id=device_id,
                    label=label,
                    kind="output",
                    is_default=index == default_output,
                    hostapi=hostapi_name,
                    max_input_channels=int(device.get("max_input_channels", 0)),
                    max_output_channels=int(device.get("max_output_channels", 0)),
                    default_sample_rate=default_sample_rate,
                ))
                output_indexes_by_id[device_id] = index
        self._annotate_wasapi_compatibility(
            inputs=inputs,
            outputs=outputs,
            devices=devices,
            hostapis=hostapis,
            input_indexes_by_id=input_indexes_by_id,
            output_indexes_by_id=output_indexes_by_id,
        )
        self._logger.info("audio devices enumerated: %s inputs, %s outputs", len(inputs), len(outputs))
        return AudioDevicesSnapshot(inputs=inputs, outputs=outputs, status="ok", error=None)

    def apply_engine_config(self, config: EngineConfig) -> AudioStatusSnapshot:
        desired = _clone_audio_config(config.audio)
        timestamp = datetime.now(timezone.utc).isoformat()
        self._logger.info(
            "apply_engine_config requested: input=%s output=%s sr=%s ch=%s",
            desired.input_device_id,
            desired.output_device_id,
            desired.sample_rate,
            desired.channels,
        )
        audio_engine_cls, passthrough_processor_cls, import_error = self._load_audio_runtime()

        if import_error is not None or audio_engine_cls is None or passthrough_processor_cls is None:
            self._logger.error("audio runtime import unavailable: %s", import_error or "Audio runtime is unavailable")
            self._status = AudioStatusSnapshot(
                status="failed",
                active_config=_clone_audio_config(self._status.active_config),
                desired_config=desired,
                last_error=import_error or "Audio runtime is unavailable",
                last_applied_at=timestamp,
            )
            return self.get_status()

        try:
            input_index = self._resolve_device_index(
                audio_engine_cls,
                desired.input_device_id,
                require_input=True,
            )
            output_index = self._resolve_device_index(
                audio_engine_cls,
                desired.output_device_id,
                require_output=True,
            )
        except LookupError as exc:
            self._logger.warning("audio device resolution failed: %s", exc)
            self._status = AudioStatusSnapshot(
                status="device_unavailable",
                active_config=_clone_audio_config(self._status.active_config),
                desired_config=desired,
                last_error=str(exc),
                last_applied_at=timestamp,
            )
            return self.get_status()
        except Exception as exc:
            self._logger.exception("audio device resolution crashed")
            self._status = AudioStatusSnapshot(
                status="failed",
                active_config=_clone_audio_config(self._status.active_config),
                desired_config=desired,
                last_error=str(exc),
                last_applied_at=timestamp,
            )
            return self.get_status()

        try:
            devices, hostapis = self._load_device_catalog(audio_engine_cls)
            self._logger.info(
                "resolved audio devices: input=%s | output=%s",
                self._describe_device(index=input_index, device=devices[input_index], hostapis=hostapis),
                self._describe_device(index=output_index, device=devices[output_index], hostapis=hostapis),
            )
            runtime_config, sample_rate_candidates = self._resolve_runtime_audio_config(
                desired=desired,
                devices=devices,
                hostapis=hostapis,
                input_index=input_index,
                output_index=output_index,
            )
            self._logger.info(
                "validated sample-rate candidates for input=%s output=%s: %s",
                desired.input_device_id,
                desired.output_device_id,
                sample_rate_candidates,
            )
        except ValueError as exc:
            self._logger.warning("audio setting validation failed: %s", exc)
            self._status = AudioStatusSnapshot(
                status="failed",
                active_config=_clone_audio_config(self._status.active_config),
                desired_config=desired,
                last_error=str(exc),
                last_applied_at=timestamp,
            )
            return self.get_status()
        except Exception as exc:
            self._logger.exception("audio setting validation crashed")
            self._status = AudioStatusSnapshot(
                status="failed",
                active_config=_clone_audio_config(self._status.active_config),
                desired_config=desired,
                last_error=str(exc),
                last_applied_at=timestamp,
            )
            return self.get_status()

        if self._engine is not None and getattr(self._engine, "_running", False) and self._same_audio_config(self._status.active_config, runtime_config):
            self._logger.info("audio engine already running with the same config; keeping active streams")
            self._status = AudioStatusSnapshot(
                status="applied",
                active_config=runtime_config,
                desired_config=desired,
                last_error=None,
                last_applied_at=timestamp,
            )
            return self.get_status()

        if self._engine is not None:
            self._logger.info("restarting running audio engine for new device settings")
            self._engine.close()

        last_start_error: str | None = None
        for candidate_sample_rate in sample_rate_candidates:
            candidate_config = replace(runtime_config, sample_rate=str(candidate_sample_rate))
            try:
                processor = self._build_processor_for_config(
                    passthrough_processor_cls=passthrough_processor_cls,
                    samplerate=int(candidate_config.sample_rate),
                    channels=int(candidate_config.channels),
                )
                next_engine = audio_engine_cls(
                    capture_device=input_index,
                    playback_device=output_index,
                    samplerate=int(candidate_config.sample_rate),
                    channels=int(candidate_config.channels),
                    processor=processor,
                )
                next_engine.start()
                self._engine = next_engine
                self._logger.info("audio engine started successfully at %s Hz", candidate_config.sample_rate)
                self._status = AudioStatusSnapshot(
                    status="applied",
                    active_config=candidate_config,
                    desired_config=desired,
                    last_error=None,
                    last_applied_at=timestamp,
                )
                return self.get_status()
            except Exception as exc:
                last_start_error = str(exc)
                self._logger.warning(
                    "audio engine start failed at %s Hz, trying next fallback if available: %s",
                    candidate_sample_rate,
                    exc,
                )
                self._engine = None

        self._status = AudioStatusSnapshot(
            status="failed",
            active_config=_clone_audio_config(self._status.active_config),
            desired_config=desired,
            last_error=last_start_error or "Audio engine start failed",
            last_applied_at=timestamp,
        )
        return self.get_status()

    def apply_active_profile(self, profile: SavedProfile | None, config: EngineConfig) -> RuntimeProfileStatusSnapshot:
        self._session_preview_status = SessionPreviewStatusSnapshot(
            session_id=None,
            target=None,
            label=None,
            processor_mode="passthrough",
            eq_curve_ready=False,
            eq_band_count=None,
            preamp_db=None,
            applied_to_audio=False,
            apply_status="idle",
            last_error=None,
        )
        if profile is None:
            self._pending_eq_profile = None
            self._logger.info("runtime profile cleared; processor mode passthrough")
            self._runtime_profile_status = RuntimeProfileStatusSnapshot(
                active_profile_id=None,
                active_profile_name=None,
                processor_mode="passthrough",
                eq_curve_ready=False,
                eq_band_count=None,
                preamp_db=None,
                applied_to_audio=self._engine is not None,
                apply_status="bypass" if not self._processing_enabled else "not_applied",
                last_error=None,
            )
            if self._engine is not None:
                self._engine.set_processor(self._create_passthrough_processor())
            return self.get_runtime_profile_status()

        try:
            eq_profile, eq_curve = self._build_eq_profile(profile)
            self._pending_eq_profile = eq_profile
            self._logger.info(
                "runtime EQ prepared for active profile %s (%s), processing_enabled=%s",
                profile.profile_id,
                profile.name,
                self._processing_enabled,
            )
            applied_to_audio = False
            apply_status = "bypass" if not self._processing_enabled else "ready"
            last_error = None
            if self._engine is not None:
                processor = self._create_passthrough_processor() if not self._processing_enabled else self._create_equalizer_processor(
                    samplerate=int(config.audio.sample_rate),
                    channels=int(config.audio.channels),
                    eq_profile=eq_profile,
                )
                self._engine.set_processor(processor)
                self._logger.info("runtime processor applied to running audio engine, mode=%s", "passthrough" if not self._processing_enabled else "eq")
                applied_to_audio = True
                apply_status = "bypass" if not self._processing_enabled else "applied"

            self._runtime_profile_status = RuntimeProfileStatusSnapshot(
                active_profile_id=profile.profile_id,
                active_profile_name=profile.name,
                processor_mode="passthrough" if not self._processing_enabled else "eq",
                eq_curve_ready=True,
                eq_band_count=int(len(eq_curve.freqs_hz)),
                preamp_db=float(eq_curve.preamp_db),
                applied_to_audio=applied_to_audio,
                apply_status=apply_status,
                last_error=last_error,
            )
            return self.get_runtime_profile_status()
        except Exception as exc:
            self._logger.exception("runtime EQ preparation failed for profile %s", profile.profile_id)
            self._runtime_profile_status = RuntimeProfileStatusSnapshot(
                active_profile_id=profile.profile_id,
                active_profile_name=profile.name,
                processor_mode="passthrough",
                eq_curve_ready=False,
                eq_band_count=None,
                preamp_db=None,
                applied_to_audio=False,
                apply_status="failed",
                last_error=str(exc),
            )
            return self.get_runtime_profile_status()

    def set_processing_enabled(self, enabled: bool, profile: SavedProfile | None, config: EngineConfig) -> RuntimeProfileStatusSnapshot:
        self._processing_enabled = bool(enabled)
        self._logger.info("processing_enabled changed: %s", self._processing_enabled)
        return self.apply_active_profile(profile, config)

    def apply_session_preview(
        self,
        *,
        session_id: str,
        target: str,
        params: PerceptualParams,
        label: str,
        config: EngineConfig,
    ) -> SessionPreviewStatusSnapshot:
        self._logger.info("session preview requested: session_id=%s target=%s label=%s", session_id, target, label)
        try:
            eq_profile, eq_curve = self._build_eq_profile_from_params(
                profile_id=f"session-preview:{session_id}:{target}",
                label=label,
                params=params,
            )
            applied_to_audio = False
            apply_status = "ready"
            if self._engine is not None:
                processor = self._create_equalizer_processor(
                    samplerate=int(config.audio.sample_rate),
                    channels=int(config.audio.channels),
                    eq_profile=eq_profile,
                )
                self._engine.set_processor(processor)
                applied_to_audio = True
                apply_status = "applied"
                self._logger.info("session preview applied to running audio engine: session_id=%s target=%s", session_id, target)
            else:
                self._logger.info("session preview prepared without live audio engine: session_id=%s target=%s", session_id, target)

            if self._runtime_profile_status.active_profile_id is not None:
                self._runtime_profile_status.applied_to_audio = False
                if self._runtime_profile_status.apply_status == "applied":
                    self._runtime_profile_status.apply_status = "preview_overridden"

            self._session_preview_status = SessionPreviewStatusSnapshot(
                session_id=session_id,
                target=target,
                label=label,
                processor_mode="eq",
                eq_curve_ready=True,
                eq_band_count=int(len(eq_curve.freqs_hz)),
                preamp_db=float(eq_curve.preamp_db),
                applied_to_audio=applied_to_audio,
                apply_status=apply_status,
                last_error=None,
            )
            return self.get_session_preview_status()
        except Exception as exc:
            self._logger.exception("session preview apply failed: session_id=%s target=%s", session_id, target)
            self._session_preview_status = SessionPreviewStatusSnapshot(
                session_id=session_id,
                target=target if target in {"A", "B", "base"} else None,
                label=label,
                processor_mode="passthrough",
                eq_curve_ready=False,
                eq_band_count=None,
                preamp_db=None,
                applied_to_audio=False,
                apply_status="failed",
                last_error=str(exc),
            )
            return self.get_session_preview_status()

    def _resolve_device_index(
        self,
        audio_engine_cls: type[Any],
        configured_device: str,
        *,
        require_input: bool = False,
        require_output: bool = False,
    ) -> int:
        devices = audio_engine_cls.list_devices()
        if configured_device.isdigit():
            direct_index = int(configured_device)
            if 0 <= direct_index < len(devices):
                device = devices[direct_index]
                if require_input and int(device.get("max_input_channels", 0)) <= 0:
                    raise LookupError(f"Configured input device is unavailable: {configured_device}")
                if require_output and int(device.get("max_output_channels", 0)) <= 0:
                    raise LookupError(f"Configured output device is unavailable: {configured_device}")
                return direct_index
        exact_match: int | None = None
        fuzzy_match: int | None = None
        configured_key = configured_device.casefold()

        for index, device in enumerate(devices):
            name = str(device.get("name", ""))
            if require_input and int(device.get("max_input_channels", 0)) <= 0:
                continue
            if require_output and int(device.get("max_output_channels", 0)) <= 0:
                continue
            name_key = name.casefold()
            if name_key == configured_key:
                exact_match = index
                break
            if configured_key and configured_key in name_key and fuzzy_match is None:
                fuzzy_match = index

        resolved = exact_match if exact_match is not None else fuzzy_match
        if resolved is None:
            raise LookupError(f"Configured audio device is unavailable: {configured_device}")
        return resolved

    def _load_device_catalog(self, audio_engine_cls: type[Any]) -> tuple[list[Any], list[Any]]:
        import sounddevice as sd

        return list(audio_engine_cls.list_devices()), list(sd.query_hostapis())

    def _resolve_hostapi_name(self, *, device: Any, hostapis: list[Any]) -> str | None:
        hostapi_index = int(device.get("hostapi", -1))
        return hostapis[hostapi_index]["name"] if 0 <= hostapi_index < len(hostapis) else None

    def _annotate_wasapi_compatibility(
        self,
        *,
        inputs: list[AudioDeviceInfo],
        outputs: list[AudioDeviceInfo],
        devices: list[Any],
        hostapis: list[Any],
        input_indexes_by_id: dict[str, int],
        output_indexes_by_id: dict[str, int],
    ) -> None:
        wasapi_inputs = [device for device in inputs if self._is_wasapi_hostapi(device.hostapi)]
        wasapi_outputs = [device for device in outputs if self._is_wasapi_hostapi(device.hostapi)]
        for input_info in wasapi_inputs:
            input_index = input_indexes_by_id.get(input_info.device_id)
            if input_index is None:
                continue
            for output_info in wasapi_outputs:
                output_index = output_indexes_by_id.get(output_info.device_id)
                if output_index is None:
                    continue
                channels = min(
                    2,
                    input_info.max_input_channels or 0,
                    output_info.max_output_channels or 0,
                )
                if channels <= 0:
                    continue
                compatible_sample_rates = self._find_pair_sample_rates(
                    input_index=input_index,
                    output_index=output_index,
                    input_device=devices[input_index],
                    output_device=devices[output_index],
                    channels=channels,
                )
                if not compatible_sample_rates:
                    continue
                input_info.compatible_device_ids.append(output_info.device_id)
                output_info.compatible_device_ids.append(input_info.device_id)
                for sample_rate in compatible_sample_rates:
                    if sample_rate not in input_info.compatible_sample_rates:
                        input_info.compatible_sample_rates.append(sample_rate)
                    if sample_rate not in output_info.compatible_sample_rates:
                        output_info.compatible_sample_rates.append(sample_rate)

    def _find_pair_sample_rates(
        self,
        *,
        input_index: int,
        output_index: int,
        input_device: Any,
        output_device: Any,
        channels: int,
    ) -> list[str]:
        sample_rates: list[str] = []
        for candidate in self._collect_sample_rate_candidates(
            requested_sample_rate=None,
            input_device=input_device,
            output_device=output_device,
        ):
            try:
                self._validate_settings(
                    input_index=input_index,
                    output_index=output_index,
                    samplerate=candidate,
                    channels=channels,
                )
                sample_rates.append(str(candidate))
            except Exception:
                continue
        return sample_rates

    def _resolve_runtime_audio_config(
        self,
        *,
        desired: EngineAudioConfig,
        devices: list[Any],
        hostapis: list[Any],
        input_index: int,
        output_index: int,
    ) -> tuple[EngineAudioConfig, list[int]]:
        input_device = devices[input_index]
        output_device = devices[output_index]
        input_hostapi = self._resolve_hostapi_name(device=input_device, hostapis=hostapis)
        output_hostapi = self._resolve_hostapi_name(device=output_device, hostapis=hostapis)
        if input_hostapi != output_hostapi:
            raise ValueError(
                f"Input/output host API mismatch: input={input_hostapi or 'unknown'}, output={output_hostapi or 'unknown'}"
            )

        requested_sample_rate = self._safe_int(desired.sample_rate)
        sample_rate_candidates = self._build_sample_rate_candidates(
            input_index=input_index,
            output_index=output_index,
            input_device=input_device,
            output_device=output_device,
            requested_sample_rate=requested_sample_rate,
            channels=self._safe_int(desired.channels),
        )
        return (
            EngineAudioConfig(
                input_device_id=desired.input_device_id,
                output_device_id=desired.output_device_id,
                sample_rate=str(sample_rate_candidates[0]),
                channels=desired.channels,
            ),
            sample_rate_candidates,
        )

    def _validate_settings(self, *, input_index: int, output_index: int, samplerate: int, channels: int) -> None:
        import sounddevice as sd

        sd.check_input_settings(device=input_index, samplerate=samplerate, channels=channels)
        sd.check_output_settings(device=output_index, samplerate=samplerate, channels=channels)

    def _probe_settings(
        self,
        *,
        input_index: int,
        output_index: int,
        samplerate: int,
        channels: int,
    ) -> tuple[bool, str | None, bool, str | None]:
        import sounddevice as sd

        input_ok = True
        input_error: str | None = None
        output_ok = True
        output_error: str | None = None

        try:
            sd.check_input_settings(device=input_index, samplerate=samplerate, channels=channels)
        except Exception as exc:
            input_ok = False
            input_error = str(exc)

        try:
            sd.check_output_settings(device=output_index, samplerate=samplerate, channels=channels)
        except Exception as exc:
            output_ok = False
            output_error = str(exc)

        return input_ok, input_error, output_ok, output_error

    def _build_sample_rate_candidates(
        self,
        *,
        input_index: int,
        output_index: int,
        input_device: Any,
        output_device: Any,
        requested_sample_rate: int,
        channels: int,
    ) -> list[int]:
        candidates = self._collect_sample_rate_candidates(
            requested_sample_rate=requested_sample_rate,
            input_device=input_device,
            output_device=output_device,
        )

        valid_candidates: list[int] = []
        candidate_failures: list[str] = []
        for candidate in candidates:
            input_ok, input_error, output_ok, output_error = self._probe_settings(
                input_index=input_index,
                output_index=output_index,
                samplerate=candidate,
                channels=channels,
            )
            if input_ok and output_ok:
                valid_candidates.append(candidate)
                continue
            candidate_failures.append(
                f"{candidate}Hz: input={'ok' if input_ok else input_error}; output={'ok' if output_ok else output_error}"
            )

        if valid_candidates:
            if valid_candidates[0] != requested_sample_rate:
                self._logger.info(
                    "sample rate fallback applied: requested=%s resolved=%s input=%s output=%s",
                    requested_sample_rate,
                    valid_candidates[0],
                    input_index,
                    output_index,
                )
            return valid_candidates

        self._logger.warning(
            "no compatible sample rate for input=%s output=%s channels=%s; candidates=%s; failures=%s",
            input_index,
            output_index,
            channels,
            candidates,
            candidate_failures,
        )

        raise ValueError(
            "No compatible sample rate found for the selected WASAPI input/output pair"
        )

    def _collect_sample_rate_candidates(
        self,
        *,
        requested_sample_rate: int | None,
        input_device: Any,
        output_device: Any,
    ) -> list[int]:
        candidates: list[int] = []
        for value in (
            requested_sample_rate,
            self._safe_int(self._normalize_sample_rate(output_device.get("default_samplerate"))),
            self._safe_int(self._normalize_sample_rate(input_device.get("default_samplerate"))),
            48000,
            44100,
            88200,
            96000,
        ):
            if value is None or value <= 0 or value in candidates:
                continue
            candidates.append(value)
        return candidates

    @staticmethod
    def _normalize_sample_rate(value: Any) -> str | None:
        if value is None:
            return None
        try:
            return str(int(float(value)))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _safe_int(value: Any) -> int | None:
        try:
            return int(str(value))
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _is_wasapi_hostapi(hostapi_name: str | None) -> bool:
        return hostapi_name == WASAPI_HOSTAPI_NAME

    def _describe_device(self, *, index: int, device: Any, hostapis: list[Any]) -> str:
        return (
            f"#{index} '{device.get('name', index)}' "
            f"hostapi={self._resolve_hostapi_name(device=device, hostapis=hostapis)} "
            f"in={int(device.get('max_input_channels', 0))} "
            f"out={int(device.get('max_output_channels', 0))} "
            f"default_sr={self._normalize_sample_rate(device.get('default_samplerate'))}"
        )

    def _load_audio_runtime(self) -> tuple[type[Any] | None, type[Any] | None, str | None]:
        try:
            from earloop.audio.engine import AudioEngine
            from earloop.audio.processors import PassthroughProcessor
        except Exception as exc:
            return None, None, str(exc)
        return AudioEngine, PassthroughProcessor, None

    def _build_processor_for_config(self, *, passthrough_processor_cls: type[Any], samplerate: int, channels: int) -> Any:
        if not self._processing_enabled or self._pending_eq_profile is None:
            return passthrough_processor_cls()
        return self._create_equalizer_processor(samplerate=samplerate, channels=channels, eq_profile=self._pending_eq_profile)

    def _create_equalizer_processor(self, *, samplerate: int, channels: int, eq_profile: Any) -> Any:
        from earloop.audio.processors import EqualizerProcessor

        return EqualizerProcessor(
            samplerate=samplerate,
            channels=channels,
            initial_profile=eq_profile,
        )

    def _create_passthrough_processor(self) -> Any:
        from earloop.audio.processors import PassthroughProcessor

        return PassthroughProcessor()

    def _build_eq_profile(self, profile: SavedProfile) -> tuple[Any, Any]:
        return self._build_eq_profile_from_params(
            profile_id=profile.profile_id,
            label=profile.name,
            params=profile.params,
        )

    def _build_eq_profile_from_params(self, *, profile_id: str, label: str, params: PerceptualParams) -> tuple[Any, Any]:
        import numpy as np
        from earloop.audio.processors import EqProfile
        from earloop.ml.mapping.parametric_eq import ParametricEqMapper
        from earloop.ml.types import PerceptualProfile

        mapper = ParametricEqMapper(np.asarray(EQ_BAND_FREQUENCIES, dtype=np.float32))
        perceptual_profile = PerceptualProfile(
            values=np.asarray([
                params.bass,
                params.tilt,
                params.presence,
                params.air,
                params.lowmid,
                params.sparkle,
            ], dtype=np.float32),
            profile_id=profile_id,
            label=label,
        )
        eq_curve = mapper.map_profile(perceptual_profile)
        eq_profile = EqProfile(
            profile_id=abs(hash(profile_id)) % (10 ** 9),
            freqs_hz=eq_curve.freqs_hz,
            gains_db=eq_curve.gains_db,
            preamp_db=float(eq_curve.preamp_db),
            name=label,
        )
        return eq_profile, eq_curve

    @staticmethod
    def _same_audio_config(left: EngineAudioConfig | None, right: EngineAudioConfig | None) -> bool:
        if left is None or right is None:
            return False
        return (
            left.input_device_id == right.input_device_id
            and left.output_device_id == right.output_device_id
            and left.sample_rate == right.sample_rate
            and left.channels == right.channels
        )
