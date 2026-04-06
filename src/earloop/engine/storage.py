from __future__ import annotations

import math
import re
from datetime import datetime, timezone
from uuid import uuid4
from dataclasses import replace
from typing import Any

import numpy as np

from earloop import __version__ as EARLOOP_VERSION
from earloop.ml.preference.linear import LinearPreferenceModel
from earloop.ml.types import PerceptualPair, PerceptualProfile, PreferenceChoice

from .persistence import (
    append_event_log_entry,
    load_or_create_user_identity,
    save_user_identity,
    load_persisted_domain_state,
    save_persisted_domain_state,
)
from .types import (
    DomainState,
    EngineAudioConfig,
    EngineConfig,
    EngineStatus,
    MainRuntimeState,
    PairData,
    PerceptualParams,
    PipelineConfig,
    SavedProfile,
    SessionSnapshot,
    SessionState,
)


DEFAULT_PROFILE_NAME = "Новый профиль"
SESSION_DEFAULT_PROFILE_ID = "__default__"
INITIAL_ITERATION = 12
INITIAL_PROGRESS = 62
PIPELINE_CATALOG = {
    "models": [
        {"id": "preference-linear-v1", "label": "Preference Linear v1"},
        {"id": "preference-hybrid-v2", "label": "Preference Hybrid v2"},
        {"id": "listener-embedding-v1", "label": "Listener Embedding v1"},
    ],
    "eqs": [
        {"id": "parametric-eq-v1", "label": "Parametric EQ v1"},
        {"id": "minimum-phase-eq-v2", "label": "Minimum Phase EQ v2"},
        {"id": "wideband-eq-v1", "label": "Wideband EQ v1"},
    ],
    "mappers": [
        {"id": "perceptual-mapper-v1", "label": "Perceptual Mapper v1"},
        {"id": "smooth-curve-mapper-v2", "label": "Smooth Curve Mapper v2"},
        {"id": "target-blend-mapper-v1", "label": "Target Blend Mapper v1"},
    ],
    "generators": [
        {"id": "candidate-pool-v1", "label": "Candidate Pool v1"},
        {"id": "local-search-v2", "label": "Local Search v2"},
        {"id": "guided-random-v1", "label": "Guided Random v1"},
    ],
    "strategies": [
        {"id": "adaptive-updater-v1", "label": "Adaptive Updater v1"},
        {"id": "confidence-weighted-v2", "label": "Confidence Weighted v2"},
        {"id": "fast-explore-v1", "label": "Fast Explore v1"},
    ],
}
STARTUP_STEPS = [
    "Инициализация аудиомоста",
    "Подключение модели персонализации",
    "Генерация первых вариантов",
]
PERCEPTUAL_DIM = 6
MODEL_VERSION = "linear_preference_v1"
EVENT_LOG_TAIL_LIMIT = 128


def _default_profile_params() -> PerceptualParams:
    return PerceptualParams(
        bass=0.18,
        tilt=-0.08,
        presence=0.24,
        air=0.16,
        lowmid=-0.12,
        sparkle=0.20,
    )


def _default_pipeline_config() -> PipelineConfig:
    return PipelineConfig(
        model_id="preference-linear-v1",
        eq_id="parametric-eq-v1",
        mapper_id="perceptual-mapper-v1",
        generator_id="candidate-pool-v1",
        strategy_id="adaptive-updater-v1",
    )


def _initial_profiles() -> list[SavedProfile]:
    return [
        SavedProfile(
            profile_id="wave",
            name="Моя волна",
            params=PerceptualParams(0.34, -0.06, 0.41, 0.22, -0.18, 0.30),
            pipeline_config=PipelineConfig("preference-linear-v1", "parametric-eq-v1", "perceptual-mapper-v1", "candidate-pool-v1", "adaptive-updater-v1"),
        ),
        SavedProfile(
            profile_id="night",
            name="Night Drive",
            params=PerceptualParams(0.52, -0.12, 0.18, 0.09, -0.22, 0.12),
            pipeline_config=PipelineConfig("preference-hybrid-v2", "minimum-phase-eq-v2", "smooth-curve-mapper-v2", "local-search-v2", "confidence-weighted-v2"),
        ),
        SavedProfile(
            profile_id="soft",
            name="Soft Focus",
            params=PerceptualParams(0.12, 0.04, -0.08, 0.28, 0.12, 0.34),
            pipeline_config=PipelineConfig("listener-embedding-v1", "wideband-eq-v1", "target-blend-mapper-v1", "guided-random-v1", "fast-explore-v1"),
        ),
    ]


def _clone_params(params: PerceptualParams) -> PerceptualParams:
    return replace(params)


def _clone_pipeline_config(config: PipelineConfig) -> PipelineConfig:
    return replace(config)


def _clone_profile(profile: SavedProfile) -> SavedProfile:
    return SavedProfile(
        profile_id=profile.profile_id,
        name=profile.name,
        params=_clone_params(profile.params),
        pipeline_config=_clone_pipeline_config(profile.pipeline_config),
    )


def _clamp(value: float, minimum: float, maximum: float) -> float:
    return max(minimum, min(maximum, value))


def _pseudo_random(seed: float) -> float:
    value = math.sin(seed * 12.9898) * 43758.5453
    return value - math.floor(value)


def _build_pair(seed: int, key: str, base_params: PerceptualParams) -> PairData:
    key_offset = 0.71 if key == "A" else 1.37

    def sample(n: float, scale: float = 1.0) -> float:
        return (_pseudo_random(seed + key_offset + n) * 2.0 - 1.0) * scale

    params = PerceptualParams(
        bass=round(_clamp(base_params.bass + sample(0.11, 0.42), -1.0, 1.0), 2),
        tilt=round(_clamp(base_params.tilt + sample(0.29, 0.24), -1.0, 1.0), 2),
        presence=round(_clamp(base_params.presence + sample(0.43, 0.38), -1.0, 1.0), 2),
        air=round(_clamp(base_params.air + sample(0.67, 0.32), -1.0, 1.0), 2),
        lowmid=round(_clamp(base_params.lowmid + sample(0.91, 0.28), -1.0, 1.0), 2),
        sparkle=round(_clamp(base_params.sparkle + sample(1.17, 0.35), -1.0, 1.0), 2),
    )
    score = round(
        params.bass * 0.42
        + params.presence * 0.36
        + params.sparkle * 0.28
        - abs(params.tilt) * 0.18
        - abs(params.lowmid) * 0.08
        + (0.24 if key == "A" else -0.06),
        2,
    )
    return PairData(pair_id=key, params=params, score=score)


def _blend_params(base_params: PerceptualParams, target_params: PerceptualParams, ratio: float) -> PerceptualParams:
    return PerceptualParams(
        bass=round(_clamp(base_params.bass + (target_params.bass - base_params.bass) * ratio, -1.0, 1.0), 2),
        tilt=round(_clamp(base_params.tilt + (target_params.tilt - base_params.tilt) * ratio, -1.0, 1.0), 2),
        presence=round(_clamp(base_params.presence + (target_params.presence - base_params.presence) * ratio, -1.0, 1.0), 2),
        air=round(_clamp(base_params.air + (target_params.air - base_params.air) * ratio, -1.0, 1.0), 2),
        lowmid=round(_clamp(base_params.lowmid + (target_params.lowmid - base_params.lowmid) * ratio, -1.0, 1.0), 2),
        sparkle=round(_clamp(base_params.sparkle + (target_params.sparkle - base_params.sparkle) * ratio, -1.0, 1.0), 2),
    )


def _apply_feedback(params: PerceptualParams, feedback: str) -> PerceptualParams:
    if feedback == "more_bass":
        return PerceptualParams(
            bass=round(_clamp(params.bass + 0.14, -1.0, 1.0), 2),
            tilt=round(_clamp(params.tilt - 0.03, -1.0, 1.0), 2),
            presence=params.presence,
            air=params.air,
            lowmid=round(_clamp(params.lowmid + 0.04, -1.0, 1.0), 2),
            sparkle=params.sparkle,
        )
    if feedback == "less_harsh":
        return PerceptualParams(
            bass=params.bass,
            tilt=round(_clamp(params.tilt - 0.08, -1.0, 1.0), 2),
            presence=round(_clamp(params.presence - 0.12, -1.0, 1.0), 2),
            air=round(_clamp(params.air - 0.04, -1.0, 1.0), 2),
            lowmid=round(_clamp(params.lowmid + 0.06, -1.0, 1.0), 2),
            sparkle=round(_clamp(params.sparkle - 0.08, -1.0, 1.0), 2),
        )
    if feedback == "more_air":
        return PerceptualParams(
            bass=params.bass,
            tilt=round(_clamp(params.tilt + 0.03, -1.0, 1.0), 2),
            presence=round(_clamp(params.presence + 0.05, -1.0, 1.0), 2),
            air=round(_clamp(params.air + 0.14, -1.0, 1.0), 2),
            lowmid=params.lowmid,
            sparkle=round(_clamp(params.sparkle + 0.07, -1.0, 1.0), 2),
        )
    if feedback == "warmer":
        return PerceptualParams(
            bass=round(_clamp(params.bass + 0.06, -1.0, 1.0), 2),
            tilt=round(_clamp(params.tilt - 0.06, -1.0, 1.0), 2),
            presence=round(_clamp(params.presence - 0.07, -1.0, 1.0), 2),
            air=round(_clamp(params.air - 0.05, -1.0, 1.0), 2),
            lowmid=round(_clamp(params.lowmid + 0.10, -1.0, 1.0), 2),
            sparkle=round(_clamp(params.sparkle - 0.04, -1.0, 1.0), 2),
        )
    return _clone_params(params)


def _params_to_vector(params: PerceptualParams) -> np.ndarray:
    return np.asarray(
        [
            params.bass,
            params.tilt,
            params.presence,
            params.air,
            params.lowmid,
            params.sparkle,
        ],
        dtype=np.float32,
    )


def _params_to_profile(params: PerceptualParams, *, profile_id: str | None = None) -> PerceptualProfile:
    return PerceptualProfile(values=_params_to_vector(params), profile_id=profile_id)


def _model_state_from_session(session: SessionState | None) -> dict[str, Any]:
    if session is not None and isinstance(session.model_state, dict) and session.model_state.get("weights"):
        return dict(session.model_state)
    return {
        "modelType": "linear_preference",
        "dim": PERCEPTUAL_DIM,
        "learningRate": 0.1,
        "l2Reg": 0.0,
        "weights": [0.0] * PERCEPTUAL_DIM,
        "version": 1,
    }


def _serialize_model_state(model: LinearPreferenceModel) -> dict[str, Any]:
    state = model.get_state()
    return {
        "modelType": state.model_type,
        "dim": state.dim,
        "learningRate": state.learning_rate,
        "l2Reg": state.l2_reg,
        "weights": [float(x) for x in state.weights],
        "version": state.version,
    }


def _build_candidate_pool(base_params: PerceptualParams, *, pair_version: int, pool_size: int = 14) -> list[PerceptualParams]:
    # candidate[0] stays close to base to keep local continuity.
    params_pool: list[PerceptualParams] = [PerceptualParams(**base_params.to_dict())]
    scales = (0.30, 0.18, 0.26, 0.24, 0.22, 0.25)
    for i in range(pool_size):
        seed = pair_version * 100 + i + 1

        def sample(axis: int, scale: float) -> float:
            return (_pseudo_random(seed + axis * 0.173) * 2.0 - 1.0) * scale

        params_pool.append(
            PerceptualParams(
                bass=round(_clamp(base_params.bass + sample(1, scales[0]), -1.0, 1.0), 2),
                tilt=round(_clamp(base_params.tilt + sample(2, scales[1]), -1.0, 1.0), 2),
                presence=round(_clamp(base_params.presence + sample(3, scales[2]), -1.0, 1.0), 2),
                air=round(_clamp(base_params.air + sample(4, scales[3]), -1.0, 1.0), 2),
                lowmid=round(_clamp(base_params.lowmid + sample(5, scales[4]), -1.0, 1.0), 2),
                sparkle=round(_clamp(base_params.sparkle + sample(6, scales[5]), -1.0, 1.0), 2),
            )
        )
    return params_pool


def _resolve_event_type(*, selected_target: str, feedback: str) -> str | None:
    if selected_target in {"A", "B"}:
        return "pair_choice"
    if selected_target == "base" and feedback != "none":
        return "directional_feedback"
    return None


def _slugify_profile_name(name: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", name.strip().lower())
    slug = slug.strip("-")
    return slug or "profile"


class InMemoryEngineStorage:
    def __init__(self) -> None:
        self._app_build_version = str(EARLOOP_VERSION)
        self._user_identity = load_or_create_user_identity()
        if not self._user_identity.get("appBuildVersion"):
            self._user_identity["appBuildVersion"] = self._app_build_version
            save_user_identity(self._user_identity)

        persisted_state = load_persisted_domain_state()
        if persisted_state is not None:
            self._state = DomainState(
                profiles=[_clone_profile(profile) for profile in persisted_state.profiles],
                config=EngineConfig(
                    audio=replace(persisted_state.config.audio),
                    active_profile_id=persisted_state.config.active_profile_id,
                    processing_enabled=persisted_state.config.processing_enabled,
                ),
                session=None,
            )
        else:
            initial_profiles = _initial_profiles()
            self._state = DomainState(
                profiles=[_clone_profile(profile) for profile in initial_profiles],
                config=EngineConfig(
                    audio=EngineAudioConfig(
                        input_device_id="CABLE Output (VB-Audio Virtual Cable)",
                        output_device_id="Динамики (Razer Barracuda X 2.4)",
                        sample_rate="48000",
                        channels="2",
                    ),
                    active_profile_id=initial_profiles[0].profile_id,
                    processing_enabled=True,
                ),
                session=None,
            )

    def _persist_domain_state(self) -> None:
        save_persisted_domain_state(
            DomainState(
                profiles=self.get_profiles(),
                config=self.get_engine_config(),
                session=None,
            )
        )

    def get_profiles(self) -> list[SavedProfile]:
        return [_clone_profile(profile) for profile in self._state.profiles]

    def get_active_profile(self) -> SavedProfile | None:
        return next(
            (_clone_profile(profile) for profile in self._state.profiles if profile.profile_id == self._state.config.active_profile_id),
            None,
        )

    def get_engine_status(self) -> EngineStatus:
        return EngineStatus(
            default_profile_name=DEFAULT_PROFILE_NAME,
            default_profile_params=_default_profile_params(),
            default_pipeline_config=_default_pipeline_config(),
            initial_iteration=INITIAL_ITERATION,
            initial_progress=INITIAL_PROGRESS,
            pipeline_catalog={key: [dict(option) for option in value] for key, value in PIPELINE_CATALOG.items()},
            session_default_profile_id=SESSION_DEFAULT_PROFILE_ID,
            startup_steps=list(STARTUP_STEPS),
        )

    def get_engine_config(self) -> EngineConfig:
        return EngineConfig(
            audio=replace(self._state.config.audio),
            active_profile_id=self._state.config.active_profile_id,
            processing_enabled=self._state.config.processing_enabled,
        )

    def get_main_state(self) -> MainRuntimeState:
        active_profile = self.get_active_profile()
        return MainRuntimeState(
            processing_enabled=self._state.config.processing_enabled,
            active_profile_id=active_profile.profile_id if active_profile is not None else None,
            active_profile_name=active_profile.name if active_profile is not None else None,
        )

    def get_domain_state(self) -> DomainState:
        session = None if self._state.session is None else replace(
            self._state.session,
            session_base_params=_clone_params(self._state.session.session_base_params),
            pipeline_config=_clone_pipeline_config(self._state.session.pipeline_config),
            current_base_params=_clone_params(self._state.session.current_base_params),
            current_pair_a=(
                None if self._state.session.current_pair_a is None
                else PairData(
                    pair_id=self._state.session.current_pair_a.pair_id,
                    params=_clone_params(self._state.session.current_pair_a.params),
                    score=self._state.session.current_pair_a.score,
                )
            ),
            current_pair_b=(
                None if self._state.session.current_pair_b is None
                else PairData(
                    pair_id=self._state.session.current_pair_b.pair_id,
                    params=_clone_params(self._state.session.current_pair_b.params),
                    score=self._state.session.current_pair_b.score,
                )
            ),
            history=[dict(entry) for entry in self._state.session.history],
            model_state=dict(self._state.session.model_state),
            event_log_tail=[dict(entry) for entry in self._state.session.event_log_tail],
        )
        return DomainState(
            profiles=self.get_profiles(),
            config=self.get_engine_config(),
            session=session,
        )

    def resolve_session_preview_target(self, *, session_id: str, target: str) -> tuple[PerceptualParams, str]:
        current_session = self._state.session
        if current_session is None:
            raise RuntimeError("No active session to preview")
        if current_session.session_id != session_id:
            raise KeyError(f"sessionId:{session_id}")

        if target == "A":
            if current_session.current_pair_a is None:
                raise RuntimeError("Session pair A is unavailable")
            return _clone_params(current_session.current_pair_a.params), "Вариант A"

        if target == "B":
            if current_session.current_pair_b is None:
                raise RuntimeError("Session pair B is unavailable")
            return _clone_params(current_session.current_pair_b.params), "Вариант B"

        return _clone_params(current_session.current_base_params), current_session.current_base_label

    def update_engine_config(self, payload: dict[str, Any]) -> EngineConfig:
        audio_payload = payload.get("audio", {})
        defaults_payload = payload.get("defaults", {})
        runtime_payload = payload.get("runtime", {})
        current = self._state.config
        next_active = str(defaults_payload.get("activeProfileId", current.active_profile_id))
        if not any(profile.profile_id == next_active for profile in self._state.profiles):
            next_active = self._state.profiles[0].profile_id if self._state.profiles else SESSION_DEFAULT_PROFILE_ID

        self._state.config = EngineConfig(
            audio=EngineAudioConfig(
                input_device_id=str(audio_payload.get("inputDeviceId", current.audio.input_device_id)),
                output_device_id=str(audio_payload.get("outputDeviceId", current.audio.output_device_id)),
                sample_rate=str(audio_payload.get("sampleRate", current.audio.sample_rate)),
                channels=str(audio_payload.get("channels", current.audio.channels)),
            ),
            active_profile_id=next_active,
            processing_enabled=bool(runtime_payload.get("processingEnabled", current.processing_enabled)),
        )
        self._persist_domain_state()
        return self.get_engine_config()

    def set_active_profile(self, *, profile_id: str) -> MainRuntimeState:
        if not any(profile.profile_id == profile_id for profile in self._state.profiles):
            raise KeyError(f"profileId:{profile_id}")
        self._state.config.active_profile_id = profile_id
        self._persist_domain_state()
        return self.get_main_state()

    def set_processing_enabled(self, *, enabled: bool) -> MainRuntimeState:
        self._state.config.processing_enabled = bool(enabled)
        self._persist_domain_state()
        return self.get_main_state()

    def update_profile(self, *, profile_id: str, name: str, params: dict[str, Any]) -> list[SavedProfile]:
        next_profiles: list[SavedProfile] = []
        next_params = PerceptualParams.from_dict(params)
        found = False
        for profile in self._state.profiles:
            if profile.profile_id == profile_id:
                found = True
                next_profiles.append(SavedProfile(
                    profile_id=profile.profile_id,
                    name=name,
                    params=next_params,
                    pipeline_config=_clone_pipeline_config(profile.pipeline_config),
                ))
            else:
                next_profiles.append(_clone_profile(profile))
        if not found:
            raise KeyError(f"profileId:{profile_id}")
        self._state.profiles = next_profiles
        self._persist_domain_state()
        return self.get_profiles()

    def delete_profile(self, *, profile_id: str) -> list[SavedProfile]:
        if not any(profile.profile_id == profile_id for profile in self._state.profiles):
            raise KeyError(f"profileId:{profile_id}")
        self._state.profiles = [profile for profile in self._state.profiles if profile.profile_id != profile_id]
        if not any(profile.profile_id == self._state.config.active_profile_id for profile in self._state.profiles):
            self._state.config.active_profile_id = self._state.profiles[0].profile_id if self._state.profiles else SESSION_DEFAULT_PROFILE_ID
        self._persist_domain_state()
        return self.get_profiles()

    def save_profile(
        self,
        *,
        name: str,
        final_choice: str,
        pair_a: dict[str, Any],
        pair_b: dict[str, Any],
        pipeline_config: dict[str, Any],
        session_base_params: dict[str, Any],
    ) -> dict[str, Any]:
        pair_a_params = PerceptualParams.from_dict(pair_a["params"])
        pair_b_params = PerceptualParams.from_dict(pair_b["params"])
        fallback_params = PerceptualParams.from_dict(session_base_params)
        chosen_params = pair_a_params if final_choice == "A" else pair_b_params if final_choice == "B" else fallback_params
        slug = _slugify_profile_name(name)
        profile_id = slug
        suffix = 1
        existing_ids = {profile.profile_id for profile in self._state.profiles}
        while profile_id in existing_ids:
            suffix += 1
            profile_id = f"{slug}-{suffix}"
        saved_profile = SavedProfile(
            profile_id=profile_id,
            name=name,
            params=chosen_params,
            pipeline_config=PipelineConfig.from_dict(pipeline_config),
        )
        self._state.profiles.append(saved_profile)
        self._state.config.active_profile_id = profile_id
        self._persist_domain_state()
        return {
            "profileId": profile_id,
            "profiles": [profile.to_dict() for profile in self.get_profiles()],
            "savedProfile": saved_profile.to_dict(),
        }

    def save_profile_from_session(
        self,
        *,
        name: str,
        final_choice: str,
    ) -> dict[str, Any]:
        current_session = self._state.session
        if current_session is None:
            raise RuntimeError("No active session to save from")

        if final_choice == "A" and current_session.current_pair_a is not None:
            chosen_params = _clone_params(current_session.current_pair_a.params)
        elif final_choice == "B" and current_session.current_pair_b is not None:
            chosen_params = _clone_params(current_session.current_pair_b.params)
        else:
            chosen_params = _clone_params(current_session.current_base_params)

        slug = _slugify_profile_name(name)
        profile_id = slug
        suffix = 1
        existing_ids = {profile.profile_id for profile in self._state.profiles}
        while profile_id in existing_ids:
            suffix += 1
            profile_id = f"{slug}-{suffix}"

        saved_profile = SavedProfile(
            profile_id=profile_id,
            name=name,
            params=chosen_params,
            pipeline_config=_clone_pipeline_config(current_session.pipeline_config),
        )
        self._state.profiles.append(saved_profile)
        self._state.config.active_profile_id = profile_id
        self._persist_domain_state()
        return {
            "profileId": profile_id,
            "profiles": [profile.to_dict() for profile in self.get_profiles()],
            "savedProfile": saved_profile.to_dict(),
        }

    def _resolve_session_components(self, payload: dict[str, Any]) -> tuple[SavedProfile | None, PipelineConfig, PerceptualParams, str, int, int, int]:
        profiles = self._state.profiles
        session_base_profile_id = str(payload["sessionBaseProfileId"])
        session_base_profile = next((profile for profile in profiles if profile.profile_id == session_base_profile_id), None)

        pipeline_config = PipelineConfig.from_dict(payload["pipelineConfig"]) if payload.get("pipelineConfig") else (
            _clone_pipeline_config(session_base_profile.pipeline_config) if session_base_profile is not None else _default_pipeline_config()
        )
        session_base_params = PerceptualParams.from_dict(payload["sessionBaseParams"]) if payload.get("sessionBaseParams") else (
            _clone_params(session_base_profile.params) if session_base_profile is not None else _default_profile_params()
        )
        session_base_label = str(payload.get("sessionBaseLabel") or (session_base_profile.name if session_base_profile is not None else "С нуля"))
        iteration = int(payload.get("iteration", INITIAL_ITERATION))
        progress = int(payload.get("progress", INITIAL_PROGRESS))
        pair_version = int(payload.get("pairVersion", 0))
        return session_base_profile, pipeline_config, session_base_params, session_base_label, iteration, progress, pair_version

    def _build_session_snapshot(
        self,
        *,
        session_id: str,
        status: str,
        session_base_profile: SavedProfile | None,
        pipeline_config: PipelineConfig,
        session_base_params: PerceptualParams,
        session_base_label: str,
        iteration: int,
        progress: int,
        pair_version: int,
        last_feedback: str,
        last_selected_target: str,
    ) -> SessionSnapshot:
        return SessionSnapshot(
            session_id=session_id,
            status=status,
            session_base_profile=_clone_profile(session_base_profile) if session_base_profile is not None else None,
            session_base_params=_clone_params(session_base_params),
            session_base_label=session_base_label,
            session_pipeline_config=_clone_pipeline_config(pipeline_config),
            iteration=iteration,
            progress=progress,
            pair_version=pair_version,
            last_feedback=last_feedback,
            last_selected_target=last_selected_target,
            pair_a=_build_pair(pair_version + 1, "A", session_base_params),
            pair_b=_build_pair(pair_version + 1, "B", session_base_params),
        )

    def _init_preference_model(self, session: SessionState | None) -> LinearPreferenceModel:
        state = _model_state_from_session(session)
        dim = int(state.get("dim", PERCEPTUAL_DIM))
        learning_rate = float(state.get("learningRate", 0.1))
        l2_reg = float(state.get("l2Reg", 0.0))
        weights = np.asarray(state.get("weights", [0.0] * dim), dtype=np.float32)
        if weights.shape[0] != dim:
            weights = np.zeros(dim, dtype=np.float32)
        return LinearPreferenceModel(
            dim=dim,
            learning_rate=learning_rate,
            l2_reg=l2_reg,
            initial_weights=weights,
        )

    def _update_model_from_last_choice(
        self,
        *,
        model: LinearPreferenceModel,
        selected_target: str,
        pair_a: PairData | None,
        pair_b: PairData | None,
    ) -> bool:
        if pair_a is None or pair_b is None:
            return False
        if selected_target == "A":
            choice = PreferenceChoice.LEFT
        elif selected_target == "B":
            choice = PreferenceChoice.RIGHT
        else:
            return False

        model.update_from_choice(
            PerceptualPair(
                left=_params_to_profile(pair_a.params, profile_id="A"),
                right=_params_to_profile(pair_b.params, profile_id="B"),
                pair_id=f"{pair_a.pair_id}-{pair_b.pair_id}",
            ),
            choice,
        )
        return True

    def _build_ml_pair(self, *, model: LinearPreferenceModel, next_base_params: PerceptualParams, pair_version: int) -> tuple[PairData, PairData]:
        pool = _build_candidate_pool(next_base_params, pair_version=pair_version)
        scored: list[tuple[PerceptualParams, float]] = []
        for idx, candidate in enumerate(pool):
            profile = _params_to_profile(candidate, profile_id=f"c{idx}")
            scored.append((candidate, float(model.score(profile))))
        scored.sort(key=lambda item: item[1], reverse=True)
        if len(scored) < 2:
            raise RuntimeError("candidate_pool_too_small")
        pair_a_params, pair_a_score = scored[0]
        pair_b_params, pair_b_score = scored[1]
        return (
            PairData(pair_id="A", params=_clone_params(pair_a_params), score=round(pair_a_score, 4)),
            PairData(pair_id="B", params=_clone_params(pair_b_params), score=round(pair_b_score, 4)),
        )

    def _append_event_log(
        self,
        *,
        event_type: str,
        session_id: str,
        pipeline_config: PipelineConfig,
        iteration: int,
        pair_version: int,
        base_params_before: PerceptualParams,
        pair_a_before: PairData | None,
        pair_b_before: PairData | None,
        selected_target: str,
        feedback: str,
        base_params_after: PerceptualParams,
        generation_mode: str,
        fallback_reason: str | None,
    ) -> dict[str, Any]:
        event = {
            "event_type": event_type,
            "user_id": str(self._user_identity.get("userId") or ""),
            "session_id": session_id,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "app_build_version": self._app_build_version,
            "model_version": MODEL_VERSION,
            "iteration": iteration,
            "pair_version": pair_version,
            "base_params_before": base_params_before.to_dict(),
            "pair_a_params": pair_a_before.params.to_dict() if pair_a_before is not None else None,
            "pair_b_params": pair_b_before.params.to_dict() if pair_b_before is not None else None,
            "selected_target": selected_target,
            "base_params_after": base_params_after.to_dict(),
            "generation_mode": generation_mode,
            "fallback_reason": fallback_reason,
            "pipeline_model_id": pipeline_config.model_id,
        }
        if event_type == "directional_feedback":
            event["feedback"] = feedback
        try:
            append_event_log_entry(event)
        except Exception as exc:
            event["event_log_error"] = str(exc)
        return event

    def create_session(self, payload: dict[str, Any]) -> SessionSnapshot:
        session_id = str(uuid4())
        session_base_profile_id = str(payload["sessionBaseProfileId"])
        session_base_profile, pipeline_config, session_base_params, session_base_label, iteration, progress, pair_version = self._resolve_session_components(payload)
        selected_target = str(payload.get("selectedTarget", "base"))
        feedback = str(payload.get("feedback", "none"))
        snapshot = self._build_session_snapshot(
            session_id=session_id,
            status="created",
            session_base_profile=session_base_profile,
            pipeline_config=pipeline_config,
            session_base_params=session_base_params,
            session_base_label=session_base_label,
            iteration=iteration,
            progress=progress,
            pair_version=pair_version,
            last_feedback=feedback,
            last_selected_target=selected_target,
        )
        self._state.session = SessionState(
            session_id=session_id,
            status="created",
            session_base_profile_id=session_base_profile_id,
            session_base_params=_clone_params(session_base_params),
            session_base_label=session_base_label,
            current_base_params=_clone_params(session_base_params),
            current_base_label=session_base_label,
            pipeline_config=_clone_pipeline_config(pipeline_config),
            iteration=iteration,
            progress=progress,
            pair_version=pair_version,
            last_feedback=feedback,
            last_selected_target=selected_target,
            current_pair_a=PairData(pair_id=snapshot.pair_a.pair_id, params=_clone_params(snapshot.pair_a.params), score=snapshot.pair_a.score),
            current_pair_b=PairData(pair_id=snapshot.pair_b.pair_id, params=_clone_params(snapshot.pair_b.params), score=snapshot.pair_b.score),
            history=[],
            model_state=_model_state_from_session(None),
            last_generation_mode="heuristic",
            last_fallback_reason=None,
            event_log_tail=[],
        )
        return snapshot

    def start_session(self, payload: dict[str, Any]) -> SessionSnapshot:
        snapshot = self.create_session(payload)
        if self._state.session is not None:
            self._state.session.status = "started"
            snapshot.status = "started"
        return snapshot

    def generate_next_pair(self, payload: dict[str, Any]) -> SessionSnapshot:
        session_base_profile_id = str(payload["sessionBaseProfileId"])
        selected_target = str(payload.get("selectedTarget", "base"))
        feedback = str(payload.get("feedback", "none"))
        session_base_profile, pipeline_config, payload_base_params, payload_base_label, iteration, progress, pair_version = self._resolve_session_components(payload)
        current_session = self._state.session
        base_params = (
            _clone_params(current_session.current_base_params)
            if current_session is not None
            else payload_base_params
        )
        base_label = current_session.current_base_label if current_session is not None else payload_base_label
        pair_a = current_session.current_pair_a if current_session is not None else None
        pair_b = current_session.current_pair_b if current_session is not None else None

        base_params_before = _clone_params(base_params)
        if selected_target == "A" and pair_a is not None:
            next_base_params = _blend_params(base_params, pair_a.params, 0.72)
            next_base_label = "вариант A"
        elif selected_target == "B" and pair_b is not None:
            next_base_params = _blend_params(base_params, pair_b.params, 0.72)
            next_base_label = "вариант B"
        else:
            next_base_params = _clone_params(base_params)
            next_base_label = base_label

        if selected_target == "base" and feedback != "none":
            next_base_params = _apply_feedback(next_base_params, feedback)
        iteration += 1
        progress = min(100, progress + 5)
        pair_version += 1
        model = self._init_preference_model(current_session)
        model_updated = self._update_model_from_last_choice(
            model=model,
            selected_target=selected_target,
            pair_a=pair_a,
            pair_b=pair_b,
        )
        generation_mode = "ml_scored"
        fallback_reason: str | None = None
        try:
            next_pair_a, next_pair_b = self._build_ml_pair(
                model=model,
                next_base_params=next_base_params,
                pair_version=pair_version,
            )
        except Exception as exc:
            generation_mode = "heuristic_fallback"
            fallback_reason = str(exc)
            next_pair_a = _build_pair(pair_version + 1, "A", next_base_params)
            next_pair_b = _build_pair(pair_version + 1, "B", next_base_params)

        snapshot = SessionSnapshot(
            session_id=current_session.session_id if current_session is not None else str(uuid4()),
            status="started",
            session_base_profile=_clone_profile(session_base_profile) if session_base_profile is not None else None,
            session_base_params=_clone_params(next_base_params),
            session_base_label=next_base_label,
            session_pipeline_config=_clone_pipeline_config(pipeline_config),
            iteration=iteration,
            progress=progress,
            pair_version=pair_version,
            last_feedback=feedback,
            last_selected_target=selected_target,
            pair_a=next_pair_a,
            pair_b=next_pair_b,
        )
        history = [*current_session.history] if current_session is not None else []
        history.append({
            "iteration": iteration,
            "selectedTarget": selected_target,
            "feedback": feedback,
            "baseLabel": next_base_label,
            "generationMode": generation_mode,
            "fallbackReason": fallback_reason,
        })
        history = history[-24:]
        event_type = _resolve_event_type(selected_target=selected_target, feedback=feedback)
        event_log_tail = [*(current_session.event_log_tail if current_session is not None else [])]
        if event_type is not None:
            event = self._append_event_log(
                event_type=event_type,
                session_id=current_session.session_id if current_session is not None else snapshot.session_id,
                pipeline_config=pipeline_config,
                iteration=iteration,
                pair_version=pair_version,
                base_params_before=base_params_before,
                pair_a_before=pair_a,
                pair_b_before=pair_b,
                selected_target=selected_target,
                feedback=feedback,
                base_params_after=next_base_params,
                generation_mode=generation_mode,
                fallback_reason=fallback_reason,
            )
            event_log_tail.append(event)
        next_model_state = _serialize_model_state(model) if model_updated else _model_state_from_session(current_session)
        event_log_tail = event_log_tail[-EVENT_LOG_TAIL_LIMIT:]
        session_id = current_session.session_id if current_session is not None else snapshot.session_id
        self._state.session = SessionState(
            session_id=session_id,
            status="started",
            session_base_profile_id=session_base_profile_id,
            session_base_params=_clone_params(next_base_params),
            session_base_label=next_base_label,
            current_base_params=_clone_params(next_base_params),
            current_base_label=next_base_label,
            pipeline_config=_clone_pipeline_config(pipeline_config),
            iteration=iteration,
            progress=progress,
            pair_version=pair_version,
            last_feedback=feedback,
            last_selected_target=selected_target,
            current_pair_a=PairData(pair_id=snapshot.pair_a.pair_id, params=_clone_params(snapshot.pair_a.params), score=snapshot.pair_a.score),
            current_pair_b=PairData(pair_id=snapshot.pair_b.pair_id, params=_clone_params(snapshot.pair_b.params), score=snapshot.pair_b.score),
            history=history,
            model_state=next_model_state,
            last_generation_mode=generation_mode,
            last_fallback_reason=fallback_reason,
            event_log_tail=event_log_tail,
        )
        return snapshot
