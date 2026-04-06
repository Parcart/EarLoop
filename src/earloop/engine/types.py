from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal


CommandName = Literal[
    "get_main_state",
    "get_engine_status",
    "list_audio_devices",
    "list_profiles",
    "preview_session_target",
    "set_active_profile",
    "set_processing_enabled",
    "save_profile",
    "save_profile_from_session",
    "update_profile",
    "delete_profile",
    "get_engine_config",
    "update_engine_config",
    "get_domain_state",
    "create_session",
    "start_session",
    "generate_next_pair",
]

PairKey = Literal["A", "B"]
ListeningTarget = Literal["base", "A", "B"]


@dataclass(slots=True)
class PerceptualParams:
    bass: float
    tilt: float
    presence: float
    air: float
    lowmid: float
    sparkle: float

    def to_dict(self) -> dict[str, float]:
        return {
            "bass": self.bass,
            "tilt": self.tilt,
            "presence": self.presence,
            "air": self.air,
            "lowmid": self.lowmid,
            "sparkle": self.sparkle,
        }

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PerceptualParams":
        return cls(
            bass=float(payload["bass"]),
            tilt=float(payload["tilt"]),
            presence=float(payload["presence"]),
            air=float(payload["air"]),
            lowmid=float(payload["lowmid"]),
            sparkle=float(payload["sparkle"]),
        )


@dataclass(slots=True)
class PipelineConfig:
    model_id: str
    eq_id: str
    mapper_id: str
    generator_id: str | None = None
    strategy_id: str | None = None

    def to_dict(self) -> dict[str, str]:
        data = {
            "modelId": self.model_id,
            "eqId": self.eq_id,
            "mapperId": self.mapper_id,
        }
        if self.generator_id is not None:
            data["generatorId"] = self.generator_id
        if self.strategy_id is not None:
            data["strategyId"] = self.strategy_id
        return data

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "PipelineConfig":
        return cls(
            model_id=str(payload["modelId"]),
            eq_id=str(payload["eqId"]),
            mapper_id=str(payload["mapperId"]),
            generator_id=str(payload["generatorId"]) if payload.get("generatorId") is not None else None,
            strategy_id=str(payload["strategyId"]) if payload.get("strategyId") is not None else None,
        )


@dataclass(slots=True)
class SavedProfile:
    profile_id: str
    name: str
    params: PerceptualParams
    pipeline_config: PipelineConfig

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.profile_id,
            "name": self.name,
            "params": self.params.to_dict(),
            "pipelineConfig": self.pipeline_config.to_dict(),
        }


@dataclass(slots=True)
class PairData:
    pair_id: PairKey
    params: PerceptualParams
    score: float

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.pair_id,
            "params": self.params.to_dict(),
            "score": self.score,
        }


@dataclass(slots=True)
class EngineAudioConfig:
    input_device_id: str
    output_device_id: str
    sample_rate: str
    channels: str

    def to_dict(self) -> dict[str, str]:
        return {
            "inputDeviceId": self.input_device_id,
            "outputDeviceId": self.output_device_id,
            "sampleRate": self.sample_rate,
            "channels": self.channels,
        }


@dataclass(slots=True)
class AudioDeviceInfo:
    device_id: str
    label: str
    kind: Literal["input", "output"]
    is_default: bool = False
    hostapi: str | None = None
    max_input_channels: int | None = None
    max_output_channels: int | None = None
    default_sample_rate: str | None = None
    compatible_device_ids: list[str] = field(default_factory=list)
    compatible_sample_rates: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "deviceId": self.device_id,
            "label": self.label,
            "kind": self.kind,
            "isDefault": self.is_default,
            "hostapi": self.hostapi,
            "maxInputChannels": self.max_input_channels,
            "maxOutputChannels": self.max_output_channels,
            "defaultSampleRate": self.default_sample_rate,
            "compatibleDeviceIds": list(self.compatible_device_ids),
            "compatibleSampleRates": list(self.compatible_sample_rates),
        }


@dataclass(slots=True)
class AudioDevicesSnapshot:
    inputs: list[AudioDeviceInfo]
    outputs: list[AudioDeviceInfo]
    status: str
    error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "inputs": [device.to_dict() for device in self.inputs],
            "outputs": [device.to_dict() for device in self.outputs],
            "status": self.status,
            "error": self.error,
        }


@dataclass(slots=True)
class AudioStatusSnapshot:
    status: str
    active_config: EngineAudioConfig | None
    desired_config: EngineAudioConfig | None
    last_error: str | None = None
    last_applied_at: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "activeConfig": self.active_config.to_dict() if self.active_config is not None else None,
            "desiredConfig": self.desired_config.to_dict() if self.desired_config is not None else None,
            "lastError": self.last_error,
            "lastAppliedAt": self.last_applied_at,
        }


@dataclass(slots=True)
class RuntimeProfileStatusSnapshot:
    active_profile_id: str | None
    active_profile_name: str | None
    processor_mode: str
    eq_curve_ready: bool
    eq_band_count: int | None = None
    preamp_db: float | None = None
    applied_to_audio: bool = False
    apply_status: str = "not_applied"
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "activeProfileId": self.active_profile_id,
            "activeProfileName": self.active_profile_name,
            "processorMode": self.processor_mode,
            "eqCurveReady": self.eq_curve_ready,
            "eqBandCount": self.eq_band_count,
            "preampDb": self.preamp_db,
            "appliedToAudio": self.applied_to_audio,
            "applyStatus": self.apply_status,
            "lastError": self.last_error,
        }


@dataclass(slots=True)
class SessionPreviewStatusSnapshot:
    session_id: str | None
    target: ListeningTarget | None
    label: str | None
    processor_mode: str
    eq_curve_ready: bool
    eq_band_count: int | None = None
    preamp_db: float | None = None
    applied_to_audio: bool = False
    apply_status: str = "idle"
    last_error: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "target": self.target,
            "label": self.label,
            "processorMode": self.processor_mode,
            "eqCurveReady": self.eq_curve_ready,
            "eqBandCount": self.eq_band_count,
            "preampDb": self.preamp_db,
            "appliedToAudio": self.applied_to_audio,
            "applyStatus": self.apply_status,
            "lastError": self.last_error,
        }


@dataclass(slots=True)
class EngineConfig:
    audio: EngineAudioConfig
    active_profile_id: str
    processing_enabled: bool = True

    def to_dict(self) -> dict[str, Any]:
        return {
            "audio": self.audio.to_dict(),
            "defaults": {
                "activeProfileId": self.active_profile_id,
            },
            "runtime": {
                "processingEnabled": self.processing_enabled,
            },
        }


@dataclass(slots=True)
class SessionState:
    session_id: str
    status: str
    session_base_profile_id: str
    session_base_params: PerceptualParams
    session_base_label: str
    current_base_params: PerceptualParams
    current_base_label: str
    pipeline_config: PipelineConfig
    iteration: int
    progress: int
    pair_version: int
    last_feedback: str = "none"
    last_selected_target: ListeningTarget = "base"
    current_pair_a: PairData | None = None
    current_pair_b: PairData | None = None
    history: list[dict[str, Any]] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "status": self.status,
            "sessionBaseProfileId": self.session_base_profile_id,
            "sessionBaseParams": self.session_base_params.to_dict(),
            "sessionBaseLabel": self.session_base_label,
            "currentBaseParams": self.current_base_params.to_dict(),
            "currentBaseLabel": self.current_base_label,
            "pipelineConfig": self.pipeline_config.to_dict(),
            "iteration": self.iteration,
            "progress": self.progress,
            "pairVersion": self.pair_version,
            "lastFeedback": self.last_feedback,
            "lastSelectedTarget": self.last_selected_target,
            "currentPairA": self.current_pair_a.to_dict() if self.current_pair_a is not None else None,
            "currentPairB": self.current_pair_b.to_dict() if self.current_pair_b is not None else None,
            "history": [dict(entry) for entry in self.history],
        }


@dataclass(slots=True)
class DomainState:
    profiles: list[SavedProfile]
    config: EngineConfig
    session: SessionState | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "profiles": [profile.to_dict() for profile in self.profiles],
            "config": self.config.to_dict(),
            "session": self.session.to_dict() if self.session is not None else None,
        }


@dataclass(slots=True)
class SessionSnapshot:
    session_id: str
    status: str
    session_base_profile: SavedProfile | None
    session_base_params: PerceptualParams
    session_base_label: str
    session_pipeline_config: PipelineConfig
    iteration: int
    progress: int
    pair_version: int
    pair_a: PairData
    pair_b: PairData
    last_feedback: str = "none"
    last_selected_target: ListeningTarget = "base"

    def to_dict(self) -> dict[str, Any]:
        return {
            "sessionId": self.session_id,
            "status": self.status,
            "sessionBaseProfile": self.session_base_profile.to_dict() if self.session_base_profile is not None else None,
            "sessionBaseParams": self.session_base_params.to_dict(),
            "sessionBaseLabel": self.session_base_label,
            "sessionPipelineConfig": self.session_pipeline_config.to_dict(),
            "iteration": self.iteration,
            "progress": self.progress,
            "pairVersion": self.pair_version,
            "lastFeedback": self.last_feedback,
            "lastSelectedTarget": self.last_selected_target,
            "pairA": self.pair_a.to_dict(),
            "pairB": self.pair_b.to_dict(),
        }


@dataclass(slots=True)
class EngineStatus:
    default_profile_name: str
    default_profile_params: PerceptualParams
    default_pipeline_config: PipelineConfig
    initial_iteration: int
    initial_progress: int
    pipeline_catalog: dict[str, list[dict[str, str]]]
    session_default_profile_id: str
    startup_steps: list[str]
    audio_status: AudioStatusSnapshot | None = None
    runtime_profile_status: RuntimeProfileStatusSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "defaultProfileName": self.default_profile_name,
            "defaultProfileParams": self.default_profile_params.to_dict(),
            "defaultPipelineConfig": self.default_pipeline_config.to_dict(),
            "initialIteration": self.initial_iteration,
            "initialProgress": self.initial_progress,
            "pipelineCatalog": self.pipeline_catalog,
            "sessionDefaultProfileId": self.session_default_profile_id,
            "startupSteps": self.startup_steps,
            "audioStatus": self.audio_status.to_dict() if self.audio_status is not None else None,
            "runtimeProfileStatus": self.runtime_profile_status.to_dict() if self.runtime_profile_status is not None else None,
        }


@dataclass(slots=True)
class MainRuntimeState:
    processing_enabled: bool
    active_profile_id: str | None
    active_profile_name: str | None
    audio_status: AudioStatusSnapshot | None = None
    runtime_profile_status: RuntimeProfileStatusSnapshot | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "processingEnabled": self.processing_enabled,
            "activeProfileId": self.active_profile_id,
            "activeProfileName": self.active_profile_name,
            "audioStatus": self.audio_status.to_dict() if self.audio_status is not None else None,
            "runtimeProfileStatus": self.runtime_profile_status.to_dict() if self.runtime_profile_status is not None else None,
        }


@dataclass(slots=True)
class ProtocolError:
    code: str
    message: str
    details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "code": self.code,
            "message": self.message,
        }
        if self.details:
            payload["details"] = self.details
        return payload
