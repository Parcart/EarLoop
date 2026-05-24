from __future__ import annotations

import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any

import numpy as np

from .audio_bridge import NullAudioBridge, RealtimeAudioBridge, is_vb_cable_capture_device, format_device_label
from .config import AppConfig
from .metrics import flatten_vector, vector_norm
from .profile_manager import ProfileManager, SavedEqProfile
from .session_logger import SessionLogger
from .strategy import StrategyPreset
from .vendor_path import ensure_vendor_path

ensure_vendor_path()
from personalization.contract_feedback import apply_feedback_to_model, apply_feedback_to_state, FEEDBACK_DELTAS
from personalization.contract_mapper import FREQS_23_DEFAULT
from personalization.contract_metrics import mapped_curve_metrics, mapped_pair_metrics, curve_metrics
from personalization.contract_pair_generator import ContractPairConfig, ContractPairGenerator
from personalization.contract_session import ContractSessionConfig, _select_pair_source, _feedback_phase_params
from personalization.preference_model import LogisticDistancePreferenceModel
from personalization.preference_update import update_state_from_choice
from personalization.state import FEATURE_NAMES_8D, init_preference_state
from .mapper_runtime import make_mapper
from .console import title, info, ok, warn, error, muted, key


@dataclass
class LiveSessionSummary:
    session_id: str
    strategy: str
    completed: bool
    saved_profile_path: str | None
    steps_completed: int
    feedback_count: int
    soft_stop_triggered: bool
    soft_stop_step: int | None
    final_rating: int | None
    final_z_contract: list[float] | None = None


@dataclass
class ChoiceHistoryEntry:
    step: int
    choice: str
    pair_source: str
    pair_source_group: str | None
    z_selected: np.ndarray
    z_a: np.ndarray
    z_b: np.ndarray
    curve_selected: np.ndarray
    curve_a: np.ndarray
    curve_b: np.ndarray
    pair_distance_z: float
    pair_distance_db_rms: float
    selected_max_abs_db: float
    response_time_sec: float


FEEDBACK_LABELS_RU: dict[str, tuple[str, str]] = {
    # Internal keys stay in English for compatibility with contract_feedback.py,
    # but users see only clear Russian wording.
    "too_much_bass": ("Баса слишком много", "уменьшим низ, станет меньше гула и давления"),
    "not_enough_bass": ("Не хватает баса", "добавим низ, звук станет плотнее"),
    "too_bright": ("Слишком ярко", "уберём верх, звук станет темнее и мягче"),
    "too_dark": ("Слишком темно", "добавим верх, звук станет ярче и открытее"),
    "too_muddy": ("Мутно / неразборчиво", "уберём нижнюю середину, станет чище"),
    "too_thin": ("Слишком тонко", "добавим тело и теплоту"),
    "more_presence": ("Хочу ближе вокал / атаку", "добавим presence, звук станет ближе"),
    "less_presence": ("Вокал / середина слишком выпирает", "уберём presence, звук станет спокойнее"),
    "more_air": ("Хочу больше воздуха", "добавим верхнюю открытость"),
    "less_air": ("Воздуха / верхов слишком много", "верх станет спокойнее"),
    "too_boomy": ("Бубнит", "уменьшим гулкий бас"),
    "too_harsh": ("Резко / утомляет", "смягчим presence и clarity"),
    "vocal_hidden": ("Вокал спрятан", "вокал станет ближе и понятнее"),
    "too_sharp_s": ("Режут С/Ш", "уменьшим сибилянты и остроту"),
    "make_weaker": ("Сделать эффект слабее", "профиль станет ближе к нейтральному"),
    "make_stronger": ("Сделать эффект сильнее", "текущий характер усилится"),
}


def _feedback_label_text(label: str) -> str:
    title, desc = FEEDBACK_LABELS_RU.get(label, (label, ""))
    return f"{title} — {desc}" if desc else title


def _print_session_commands() -> None:
    print("\nКоманды во время сессии:")
    print("  1/playa/aud_a  - включить A")
    print("  2/playb/aud_b  - включить B")
    print("  a              - выбрать A")
    print("  b              - выбрать B")
    print("  f              - Directional Feedback, если оба плохие")
    print("  p              - текущий лучший профиль")
    print("  n              - neutral EQ")
    print("  v              - показать z-векторы")
    print("  h/history      - прослушать прошлые выбранные варианты")
    print("  s              - выбрать, какой профиль сохранить")
    print("  q              - выйти без сохранения")


def _ask(prompt: str, default: str | None = None) -> str:
    suffix = f" [{default}]" if default is not None else ""
    try:
        value = input(f"{prompt}{suffix}: ").strip()
    except EOFError:
        return default or ""
    if not value and default is not None:
        return default
    return value


def _print_z(title: str, z) -> None:
    z = np.asarray(z, dtype=float)
    print(f"\n{title}")
    for name, value in zip(FEATURE_NAMES_8D, z):
        print(f"  {name:>11}: {value: .3f}")


def _is_soft_stop_candidate(step: int, cfg: ContractSessionConfig, state, update_norm: float, recent_update_norms: list[float]) -> bool:
    if step < int(cfg.min_ready_step):
        return False
    mean_std = float(np.mean(state.z_std))
    std_ok = mean_std <= float(cfg.ready_mean_std_threshold)
    update_ok = float(update_norm) <= float(cfg.ready_update_norm_threshold)
    plateau_ok = False
    if len(recent_update_norms) >= 4:
        plateau_ok = float(np.mean(recent_update_norms[-4:])) <= float(cfg.ready_update_norm_threshold) * 0.72
    return bool(std_ok and (update_ok or plateau_ok))


class LivePersonalizationSession:
    def __init__(
        self,
        *,
        config: AppConfig,
        strategy: StrategyPreset,
        audio: RealtimeAudioBridge | NullAudioBridge,
        data_root: Path,
        initial_z: np.ndarray | None = None,
    ) -> None:
        self.config = config
        self.strategy = strategy
        self.audio = audio
        self.data_root = Path(data_root)
        self.mapper = make_mapper(config)
        self.mapper_version = str(getattr(self.mapper, "mapper_version", self.mapper.__class__.__name__))
        if "interpretable" in self.mapper_version.lower():
            print(warn(f"[mapper] WARNING: {self.mapper_version} (fallback / not learned model)"))
        else:
            print(ok(f"[mapper] OK: {self.mapper_version}"))
        self.profile_manager = ProfileManager(self.data_root)
        self.logger = SessionLogger(self.data_root, strategy.name)
        self.session_cfg = ContractSessionConfig(
            strategy=strategy.contract_strategy,
            experiment_label=strategy.name,
            n_steps=int(config.budget_steps),
            seed=int(time.time()) % 999999,
            feedback_phase_aware=bool(strategy.phase_aware_feedback),
        )
        self.rng = np.random.default_rng(self.session_cfg.seed)
        pair_cfg = self.session_cfg.pair_config or ContractPairConfig(clip_value=self.session_cfg.clip_value)
        self.pair_generator = ContractPairGenerator(config=pair_cfg, mapper=self.mapper, rng=self.rng)
        self.state = init_preference_state(dim=8, init_std=self.session_cfg.init_std)
        if initial_z is not None:
            self.state.z_mean = np.asarray(initial_z, dtype=np.float64).copy()
        self.model = LogisticDistancePreferenceModel(
            dim=8,
            lr=self.session_cfg.model_lr,
            temperature=self.session_cfg.model_temperature,
            l2=self.session_cfg.model_l2,
            clip_value=self.session_cfg.clip_value,
        )
        self.ready_step: int | None = None
        self.pending_feedback_recovery: dict[str, Any] | None = None
        self.feedback_count = 0
        self.recent_update_norms: list[float] = []
        self.soft_stop_decisions: list[dict[str, Any]] = []
        self.last_pair: tuple[np.ndarray, np.ndarray] | None = None
        self.last_curves: tuple[np.ndarray, np.ndarray] | None = None
        self.choice_history: list[ChoiceHistoryEntry] = []
        self._aux_saved_profile_path: str | None = None
        self._last_saved_profile_path: str | None = None
        self._last_saved_profile_source: str | None = None
        self._last_saved_source_step: int | None = None
        self._last_saved_source_choice: str | None = None
        self._last_saved_z_contract: np.ndarray | None = None
        self._last_saved_eq23_db: np.ndarray | None = None
        self._active_profile: dict[str, Any] | None = None

    def _make_pair(self, step: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[str, Any], str]:
        pair_source = _select_pair_source(
            step=step,
            cfg=self.session_cfg,
            ready_step=self.ready_step,
            distances_z=[],
            distances_db_rms=[],
            rng=self.rng,
        )
        if self.pending_feedback_recovery is not None:
            pair_source = "feedback_recovery"

        if pair_source == "feedback_recovery":
            z_a, z_b, direction, pair_meta = self.pair_generator.feedback_recovery(
                state=self.state,
                feedback_label=str(self.pending_feedback_recovery.get("feedback_label")),
                strength=float(self.pending_feedback_recovery.get("feedback_strength", 1.0)),
            )
            self.pending_feedback_recovery = None
        elif pair_source == "direct":
            z_a, z_b, direction, pair_meta = self.pair_generator.direct_refinement(
                state=self.state,
                model=self.model,
                mode=self.session_cfg.direct_mode,
            )
        elif pair_source == "candidate_pool":
            z_a, z_b, direction, pair_meta = self.pair_generator.candidate_pool(self.state)
        elif pair_source == "axis":
            z_a, z_b, direction, pair_meta = self.pair_generator.axis_refinement(self.state)
        else:
            z_a, z_b, direction, pair_meta = self.pair_generator.semantic_active(self.state)
            pair_source = "semantic"

        pair_meta = dict(pair_meta)
        pair_meta["strategy"] = self.strategy.name
        pair_meta["step"] = int(step)
        pair_meta["ready_active"] = self.ready_step is not None
        pair_meta["ready_step"] = self.ready_step
        return z_a, z_b, direction, pair_meta, pair_source

    def _set_active_profile(
        self,
        *,
        label: str,
        source: str,
        z_contract: np.ndarray | None,
        curve: np.ndarray | None,
        step: int | None = None,
        choice: str | None = None,
        source_note: str | None = None,
    ) -> None:
        self._active_profile = {
            "label": label,
            "source": source,
            "z_contract": None if z_contract is None else np.asarray(z_contract, dtype=np.float64).copy(),
            "curve": None if curve is None else np.asarray(curve, dtype=np.float64).copy(),
            "step": step,
            "choice": choice,
            "source_note": source_note,
        }

    def _apply_pair_label(self, label: str, *, step: int | None = None) -> None:
        label_u = str(label).upper()
        if not self.last_pair or not self.last_curves:
            return
        if label_u == "A":
            z = self.last_pair[0]
            curve = self.last_curves[0]
        elif label_u == "B":
            z = self.last_pair[1]
            curve = self.last_curves[1]
        else:
            return
        self.audio.apply_eq(label_u, FREQS_23_DEFAULT, curve, preamp_db=self.config.preamp_db)
        self._set_active_profile(
            label=label_u,
            source=f"current_pair_{label_u}",
            z_contract=z,
            curve=curve,
            step=step,
            choice=label_u,
            source_note=f"currently playing candidate {label_u} from current A/B pair",
        )

    def _apply_label(self, label: str) -> None:
        if label.upper() in {"A", "B"}:
            self._apply_pair_label(label)
        elif label.lower() in {"neutral", "n"}:
            self.audio.apply_neutral()
            self._set_active_profile(label="neutral", source="neutral", z_contract=None, curve=None)
        elif label.lower() in {"best", "p"}:
            curve = self.mapper.map_one(self.state.z_mean)
            self.audio.apply_eq("current_best", FREQS_23_DEFAULT, curve, preamp_db=self.config.preamp_db)
            self._set_active_profile(
                label="current_best",
                source="current_state_best",
                z_contract=self.state.z_mean,
                curve=curve,
                source_note="current internal preference state / z_mean",
            )

    def _feedback_menu(self) -> tuple[str, float] | None:
        labels = list(FEEDBACK_DELTAS.keys()) + ["make_weaker", "make_stronger"]
        print("\nОбратная связь: что не так со звуком?")
        print("Выберите самый близкий вариант. Внутренние technical labels пользователю не показываются.")
        for idx, label in enumerate(labels, 1):
            print(f"[{idx:>2}] {_feedback_label_text(label)}")
        print("[0] отмена")
        raw = _ask("Выберите пункт", "0")
        if not raw.isdigit():
            return None
        idx = int(raw)
        if idx <= 0 or idx > len(labels):
            return None
        label = labels[idx - 1]
        print(f"Выбрано: {_feedback_label_text(label)}")
        strength_raw = _ask("Степень влияния 0.5..1.5, Enter = 1", "1")
        try:
            strength = float(str(strength_raw).replace(",", "."))
        except ValueError:
            strength = 1.0
        strength = float(np.clip(strength, 0.3, 1.8))
        print(f"Сила feedback: {strength:.2f}")
        return label, strength

    def _handle_soft_stop(self, step: int) -> bool:
        print("\n=== Soft-stop marker ===")
        print("Система считает, что профиль уже стабилизировался.")
        print("[1] Сохранить профиль и завершить")
        print("[2] Продолжить персонализацию")
        print("[3] Прослушать текущий профиль")
        print("[4] Сравнить текущий профиль с нейтральным EQ")
        print("[5] Прослушать / сохранить один из прошлых выбранных вариантов")
        choice = _ask("Ваш выбор", "2")
        self.logger.event("soft_stop_prompt", step=step, user_choice=choice, z_mean=self.state.z_mean, z_std=self.state.z_std)
        self.soft_stop_decisions.append({"step": step, "choice": choice})
        if choice == "1":
            path = self.save_profile_menu(finished_after_soft_stop=True)
            if path is not None:
                self._aux_saved_profile_path = str(path)
                return True
            return False
        if choice == "3":
            self._apply_label("best")
            return False
        if choice == "4":
            self._apply_label("best")
            input("Сейчас играет текущий профиль. Нажмите Enter для нейтрального EQ...")
            self._apply_label("neutral")
            input("Сейчас нейтральный EQ. Нажмите Enter чтобы продолжить...")
            return False
        if choice == "5":
            path = self._review_history_menu()
            if path is not None:
                return True
            return False
        return False

    def _append_choice_history(
        self,
        *,
        step: int,
        choice: str,
        pair_source: str,
        pair_meta: dict[str, Any],
        z_a: np.ndarray,
        z_b: np.ndarray,
        curve_a: np.ndarray,
        curve_b: np.ndarray,
        response_time: float,
    ) -> None:
        choice_u = str(choice).upper()
        if choice_u == "A":
            z_selected = z_a
            curve_selected = curve_a
        elif choice_u == "B":
            z_selected = z_b
            curve_selected = curve_b
        else:
            return
        selected_metrics = curve_metrics(curve_selected)
        self.choice_history.append(ChoiceHistoryEntry(
            step=int(step),
            choice=choice_u,
            pair_source=str(pair_source),
            pair_source_group=pair_meta.get("source_group"),
            z_selected=np.asarray(z_selected, dtype=np.float64).copy(),
            z_a=np.asarray(z_a, dtype=np.float64).copy(),
            z_b=np.asarray(z_b, dtype=np.float64).copy(),
            curve_selected=np.asarray(curve_selected, dtype=np.float64).copy(),
            curve_a=np.asarray(curve_a, dtype=np.float64).copy(),
            curve_b=np.asarray(curve_b, dtype=np.float64).copy(),
            pair_distance_z=float(pair_meta.get("pair_distance_z", np.nan)),
            pair_distance_db_rms=float(pair_meta.get("pair_distance_db_rms", np.nan)),
            selected_max_abs_db=float(getattr(selected_metrics, "max_abs_db", np.nan)),
            response_time_sec=float(response_time),
        ))

    def _print_choice_history(self) -> None:
        if not self.choice_history:
            print("\nИстория выборов пока пустая: ещё не было подтверждённых A/B-выборов.")
            return
        print("\nИстория подтверждённых выборов:")
        for idx, item in enumerate(self.choice_history, 1):
            group = item.pair_source_group or item.pair_source
            print(
                f"[{idx:>2}] step {item.step:>2} | выбрано {item.choice} | "
                f"source={group} | pair_z={item.pair_distance_z:.3f} | "
                f"pair_db_rms={item.pair_distance_db_rms:.3f} | "
                f"selected_max={item.selected_max_abs_db:.2f} dB"
            )

    def _review_history_menu(self) -> Path | None:
        if not self.choice_history:
            print("\nИстория выборов пока пустая.")
            return None

        while True:
            self._print_choice_history()
            print("[0] назад к текущей A/B-паре")
            raw = _ask("Какой прошлый шаг открыть", "0").strip().lower()
            if raw in {"", "0", "back", "b"}:
                return None
            if not raw.isdigit():
                print("Введите номер из списка или 0 для возврата.")
                continue
            idx = int(raw)
            if idx < 1 or idx > len(self.choice_history):
                print("Нет такого номера в истории.")
                continue

            item = self.choice_history[idx - 1]
            while True:
                print(f"\nStep {item.step}, выбран вариант {item.choice}")
                print("[1] прослушать выбранный вариант")
                print("[a] прослушать вариант A того шага")
                print("[b] прослушать вариант B того шага")
                print("[n] neutral EQ")
                print("[v] показать z выбранного варианта")
                print("[s] сохранить этот прошлый выбор как итоговый профиль и завершить")
                print("[0] назад к списку истории")
                cmd = _ask("Действие", "1").strip().lower()
                if cmd in {"0", "back"}:
                    break
                if cmd == "1":
                    self.audio.apply_eq(
                        f"history_step_{item.step}_{item.choice}",
                        FREQS_23_DEFAULT,
                        item.curve_selected,
                        preamp_db=self.config.preamp_db,
                    )
                    self._set_active_profile(
                        label=f"history_step_{item.step}_{item.choice}",
                        source="history_selected_choice",
                        z_contract=item.z_selected,
                        curve=item.curve_selected,
                        step=item.step,
                        choice=item.choice,
                        source_note=f"currently playing selected historical choice: step={item.step}, choice={item.choice}",
                    )
                    print(f"Сейчас играет прошлый выбранный вариант: step {item.step}, {item.choice}")
                elif cmd == "a":
                    self.audio.apply_eq(f"history_step_{item.step}_A", FREQS_23_DEFAULT, item.curve_a, preamp_db=self.config.preamp_db)
                    self._set_active_profile(
                        label=f"history_step_{item.step}_A",
                        source="history_pair_A",
                        z_contract=item.z_a,
                        curve=item.curve_a,
                        step=item.step,
                        choice="A",
                        source_note=f"currently playing historical A: step={item.step}",
                    )
                    print(f"Сейчас играет прошлый A: step {item.step}")
                elif cmd == "b":
                    self.audio.apply_eq(f"history_step_{item.step}_B", FREQS_23_DEFAULT, item.curve_b, preamp_db=self.config.preamp_db)
                    self._set_active_profile(
                        label=f"history_step_{item.step}_B",
                        source="history_pair_B",
                        z_contract=item.z_b,
                        curve=item.curve_b,
                        step=item.step,
                        choice="B",
                        source_note=f"currently playing historical B: step={item.step}",
                    )
                    print(f"Сейчас играет прошлый B: step {item.step}")
                elif cmd == "n":
                    self.audio.apply_neutral()
                elif cmd == "v":
                    _print_z(f"z_selected | step {item.step} {item.choice}", item.z_selected)
                elif cmd == "s":
                    path = self.save_profile_interactive(
                        finished_after_soft_stop=self.ready_step is not None,
                        z_contract=item.z_selected,
                        curve=item.curve_selected,
                        name_default=f"profile_{self.logger.session_id[-8:]}_step_{item.step}_{item.choice}",
                        source_note=f"saved from previous choice: step={item.step}, choice={item.choice}, source={item.pair_source_group or item.pair_source}",
                        profile_source="history_selected_choice",
                        source_step=item.step,
                        source_choice=item.choice,
                    )
                    self.logger.event(
                        "historical_profile_saved",
                        path=str(path),
                        source_step=item.step,
                        source_choice=item.choice,
                        source_group=item.pair_source_group,
                    )
                    self._aux_saved_profile_path = str(path)
                    return path
                else:
                    print("Неизвестная команда.")

    def _last_selected_entry(self) -> ChoiceHistoryEntry | None:
        return self.choice_history[-1] if self.choice_history else None

    def _profile_candidate_from_last_selected(self) -> dict[str, Any] | None:
        item = self._last_selected_entry()
        if item is None:
            return None
        return {
            "label": f"last_selected_step_{item.step}_{item.choice}",
            "source": "last_selected_choice",
            "z_contract": item.z_selected,
            "curve": item.curve_selected,
            "step": item.step,
            "choice": item.choice,
            "name_default": f"profile_{self.logger.session_id[-8:]}_last_step_{item.step}_{item.choice}",
            "source_note": f"saved from last selected choice: step={item.step}, choice={item.choice}, source={item.pair_source_group or item.pair_source}",
        }

    def _profile_candidate_from_active(self) -> dict[str, Any] | None:
        active = self._active_profile
        if not active:
            return None
        z = active.get("z_contract")
        curve = active.get("curve")
        if z is None or curve is None:
            return None
        label = str(active.get("label") or "active")
        step = active.get("step")
        choice = active.get("choice")
        return {
            "label": label,
            "source": str(active.get("source") or "currently_playing"),
            "z_contract": z,
            "curve": curve,
            "step": step,
            "choice": choice,
            "name_default": f"profile_{self.logger.session_id[-8:]}_playing_{label}",
            "source_note": str(active.get("source_note") or f"saved from currently playing profile: {label}"),
        }

    def _profile_candidate_from_state(self) -> dict[str, Any]:
        curve = self.mapper.map_one(self.state.z_mean)
        return {
            "label": "current_state_best",
            "source": "current_state_best",
            "z_contract": self.state.z_mean,
            "curve": curve,
            "step": int(self.state.step),
            "choice": None,
            "name_default": f"profile_{self.logger.session_id[-8:]}_state",
            "source_note": "saved from internal current preference state / z_mean",
        }

    def _same_profile_candidate(self, a: dict[str, Any] | None, b: dict[str, Any] | None) -> bool:
        if not isinstance(a, dict) or not isinstance(b, dict):
            return False
        za = a.get("z_contract")
        zb = b.get("z_contract")
        if za is None or zb is None:
            return False
        try:
            return bool(np.allclose(np.asarray(za, dtype=np.float64), np.asarray(zb, dtype=np.float64), atol=1e-9, rtol=1e-7))
        except Exception:
            return False

    def _preview_profile_candidate(self, slot: str, candidate: dict[str, Any]) -> None:
        z = candidate.get("z_contract")
        curve = candidate.get("curve")
        if z is None or curve is None:
            print("Этот вариант нельзя прослушать: нет EQ-кривой.")
            return
        label = str(candidate.get("label") or f"save_option_{slot}")
        source = str(candidate.get("source") or "save_menu_preview")
        self.audio.apply_eq(
            f"save_preview_{slot}_{label}",
            FREQS_23_DEFAULT,
            np.asarray(curve, dtype=np.float64),
            preamp_db=self.config.preamp_db,
        )
        self._set_active_profile(
            label=label,
            source=source,
            z_contract=np.asarray(z, dtype=np.float64),
            curve=np.asarray(curve, dtype=np.float64),
            step=candidate.get("step"),
            choice=candidate.get("choice"),
            source_note=str(candidate.get("source_note") or f"previewed from save menu option {slot}: {source}"),
        )
        self.logger.event(
            "save_menu_profile_preview",
            slot=slot,
            source=source,
            source_step=candidate.get("step"),
            source_choice=candidate.get("choice"),
            z_contract=np.asarray(z, dtype=np.float64),
            eq23_db=np.asarray(curve, dtype=np.float64),
        )
        print(f"Сейчас играет вариант [{slot}]: {source}, {label}")

    def save_profile_menu(self, finished_after_soft_stop: bool | None = None) -> Path | None:
        """Ask explicitly what should be saved.

        UX default is the last selected A/B candidate, because that is the last
        profile the user consciously accepted. The internal z_mean is still
        available as a separate research/system profile option.
        The user can also preview/listen to every save candidate before saving.
        """
        while True:
            last = self._profile_candidate_from_last_selected()
            active = self._profile_candidate_from_active()
            state = self._profile_candidate_from_state()

            print("\nЧто сохранить?")
            options: dict[str, dict[str, Any] | str] = {}
            preview_options: dict[str, dict[str, Any]] = {}
            if last is not None:
                print(f"[1] Последний ВЫБРАННЫЙ вариант: step {last['step']}, {last['choice']}  ← по умолчанию")
                print("    Это последний вариант, за который пользователь нажал A/B.")
                options["1"] = last
                preview_options["1"] = last
                default = "1"
            else:
                print("[1] Последний ВЫБРАННЫЙ вариант: пока нет A/B-выбора")
                default = "3"

            if active is not None:
                same_note = ""
                if self._same_profile_candidate(active, last):
                    same_note = " — сейчас совпадает с [1]"
                print(f"[2] Сейчас ПРОСЛУШИВАЕМЫЙ вариант: {active['label']} ({active['source']}){same_note}")
                print("    Это то, что было последним включено через 1/2/p/history.")
                options["2"] = active
                preview_options["2"] = active
            else:
                print("[2] Сейчас ПРОСЛУШИВАЕМЫЙ вариант: нет активного A/B/EQ-кандидата")

            print("[3] Расчётный профиль системы / z_mean")
            print("    Это сглаженная оценка preference state; может отличаться от услышанных A/B.")
            print("[4] Открыть историю прошлых выборов")
            print("[0] Отмена")
            options["3"] = state
            options["4"] = "history"
            preview_options["3"] = state

            print("\nПрослушать перед сохранением: l1 / l2 / l3")
            print("Сохранить: 1 / 2 / 3. По Enter сохранится вариант по умолчанию.")

            raw = _ask("Ваш выбор", default).strip().lower().replace(" ", "")
            if raw in {"0", "cancel", "отмена"}:
                return None
            if raw in {"4", "h", "history", "история"}:
                return self._review_history_menu()

            # Preview commands: l1, listen1, play1, p1, послушать1.
            preview_slot = None
            for prefix in ("l", "listen", "play", "p", "слушать", "послушать"):
                if raw.startswith(prefix):
                    preview_slot = raw[len(prefix):]
                    break
            if preview_slot in preview_options:
                self._preview_profile_candidate(preview_slot, preview_options[preview_slot])
                continue
            if raw in {"l", "listen", "play", "послушать"}:
                # Convenience: preview the default save candidate.
                candidate = preview_options.get(default)
                if candidate is not None:
                    self._preview_profile_candidate(default, candidate)
                else:
                    print("Нет варианта по умолчанию для прослушивания.")
                continue

            selected = options.get(raw)
            if selected == "history":
                return self._review_history_menu()
            if not isinstance(selected, dict):
                print("Не удалось выбрать профиль для сохранения. Используйте 1/2/3 или l1/l2/l3 для прослушивания.")
                continue
            print(f"Сохраняется источник: {selected['source']}")
            return self.save_profile_interactive(
                finished_after_soft_stop=finished_after_soft_stop,
                z_contract=selected["z_contract"],
                curve=selected["curve"],
                name_default=selected.get("name_default"),
                source_note=selected.get("source_note"),
                profile_source=selected.get("source"),
                source_step=selected.get("step"),
                source_choice=selected.get("choice"),
            )

    def _ask_rating(self) -> int | None:
        while True:
            rating_raw = _ask("Оценка профиля 1..5, можно пусто", "")
            if not rating_raw:
                return None
            try:
                rating = int(rating_raw)
            except ValueError:
                print("Оценка должна быть числом от 1 до 5 или пустой строкой.")
                continue
            if 1 <= rating <= 5:
                return rating
            print("Оценка должна быть в диапазоне 1..5. Например: 4")

    def save_profile_interactive(
        self,
        finished_after_soft_stop: bool | None = None,
        *,
        z_contract: np.ndarray | None = None,
        curve: np.ndarray | None = None,
        name_default: str | None = None,
        source_note: str | None = None,
        profile_source: str | None = None,
        source_step: int | None = None,
        source_choice: str | None = None,
    ) -> Path:
        if z_contract is None:
            z_contract = self.state.z_mean
        z_contract = np.asarray(z_contract, dtype=np.float64)
        if curve is None:
            curve = self.mapper.map_one(z_contract)
        curve = np.asarray(curve, dtype=np.float64)
        if name_default is None:
            name_default = f"profile_{self.logger.session_id[-8:]}"
        if profile_source is None:
            profile_source = "current_state_best"
        name = _ask("Название профиля", name_default)
        rating = self._ask_rating()
        comment = _ask("Комментарий, можно пусто", "")
        if comment == "":
            comment = None
        if source_note:
            comment = f"{source_note}; {comment}" if comment else source_note

        last = self._last_selected_entry()
        final_state_curve = self.mapper.map_one(self.state.z_mean)
        profile = SavedEqProfile.create(
            name=name,
            strategy=self.strategy.name,
            mapper_version=self.mapper_version,
            z_contract=z_contract,
            eq23_db=curve,
            freqs_23=FREQS_23_DEFAULT,
            session_id=self.logger.session_id,
            steps_count=int(self.state.step),
            feedback_count=int(self.feedback_count),
            soft_stop_triggered=self.ready_step is not None,
            user_finished_after_soft_stop=finished_after_soft_stop,
            final_rating=rating,
            comment=comment,
            profile_source=profile_source,
            source_step=source_step,
            source_choice=source_choice,
            final_state_z_contract=self.state.z_mean,
            final_state_eq23_db=final_state_curve,
            last_selected_step=None if last is None else last.step,
            last_selected_choice=None if last is None else last.choice,
            last_selected_z_contract=None if last is None else last.z_selected,
            last_selected_eq23_db=None if last is None else last.curve_selected,
            saved_at_step=int(self.state.step),
        )
        path = self.profile_manager.save(profile)
        self._last_saved_profile_path = str(path)
        self._last_saved_profile_source = str(profile_source)
        self._last_saved_source_step = None if source_step is None else int(source_step)
        self._last_saved_source_choice = None if source_choice is None else str(source_choice)
        self._last_saved_z_contract = z_contract.copy()
        self._last_saved_eq23_db = curve.copy()
        self.logger.event(
            "profile_saved",
            path=str(path),
            profile_id=profile.profile_id,
            rating=rating,
            comment=comment,
            profile_source=profile_source,
            source_step=source_step,
            source_choice=source_choice,
            saved_z_contract=z_contract,
            saved_eq23_db=curve,
            final_state_z_contract=self.state.z_mean,
            final_state_eq23_db=final_state_curve,
            last_selected_step=None if last is None else last.step,
            last_selected_choice=None if last is None else last.choice,
            last_selected_z_contract=None if last is None else last.z_selected,
            last_selected_eq23_db=None if last is None else last.curve_selected,
        )
        print(f"Saved profile: {path}")
        return path

    def _post_bridge_vbcable_prompt(self) -> None:
        """Prompt the tester to switch Windows Output after the bridge is already running.

        On some Windows/MME setups the most reliable order is:
        1) keep Windows Output on real headphones while starting the bridge;
        2) start the bridge;
        3) switch Windows Output to CABLE Input.
        This does not change state/model/metrics; it is only a routing helper.
        """
        if isinstance(self.audio, NullAudioBridge):
            return
        audio_cfg = getattr(self.config, "audio", None)
        if not bool(getattr(audio_cfg, "post_bridge_vb_switch_prompt", True)):
            return
        allowed = getattr(audio_cfg, "allowed_host_apis", None)
        capture = getattr(audio_cfg, "capture_device", None)
        if not is_vb_cable_capture_device(capture, allowed_host_apis=allowed):
            return
        print("\n" + title("[audio] VB-Cable routing step"))
        print("Аудиомост уже запущен.")
        print("Если вы тестируете через VB-Cable, теперь переключите системный вывод Windows:")
        print(f"  {key('Windows Output / Вывод Windows')} -> CABLE Input (VB-Audio Virtual Cable)")
        print(f"  {key('CLI capture/input')}              -> {format_device_label(capture, allowed_host_apis=allowed)}")
        print("  CLI playback/out                        -> ваши наушники/колонки")
        print("\nЕсли звук уже идёт как надо, просто нажмите Enter.")
        input("После переключения Windows Output на CABLE Input нажмите Enter...")

    def run(self) -> LiveSessionSummary:
        self.logger.event("session_started", strategy=self.strategy.name, config=asdict(self.config), mapper_version=self.mapper_version)
        saved_path: str | None = None
        completed = False
        final_rating = None
        try:
            audio_was_running = bool(getattr(self.audio, "running", False))
            if audio_was_running:
                print("[audio] bridge already running; reuse it for this session")
            self.audio.start()
            if not audio_was_running:
                self._post_bridge_vbcable_prompt()
            _print_session_commands()

            for step in range(1, int(self.config.budget_steps) + 1):
                state_before = self.state.copy()
                z_a, z_b, direction, pair_meta, pair_source = self._make_pair(step)
                curve_a, curve_b = self.mapper.map_batch(np.stack([z_a, z_b], axis=0))
                self.last_pair = (z_a, z_b)
                self.last_curves = (curve_a, curve_b)
                self.logger.log_eq_curve(step, "A", FREQS_23_DEFAULT, curve_a)
                self.logger.log_eq_curve(step, "B", FREQS_23_DEFAULT, curve_b)
                self._apply_pair_label("A", step=step)

                pair_start = time.time()
                observed_action = None
                choice = None
                feedback_label = None
                feedback_strength = 0.0
                model_record = {"loss_before": np.nan, "loss_after": np.nan, "p_before": np.nan, "p_after": np.nan}
                feedback_model_record = {"model_delta_norm": 0.0}

                print("\n" + title(f"--- Step {step}/{self.config.budget_steps} | source={pair_source} ---"))
                print(info(f"pair_distance_z={pair_meta.get('pair_distance_z', np.nan):.3f}, db_rms={pair_meta.get('pair_distance_db_rms', np.nan):.3f}, safety={pair_meta.get('safety_ok', True)}"))
                _print_session_commands()
                while True:
                    cmd = _ask("command", "")
                    cmd_l = cmd.strip().lower()
                    if cmd_l in {"1", "playa", "aud_a", "listen_a"}:
                        self._apply_pair_label("A", step=step)
                    elif cmd_l in {"2", "playb", "aud_b", "listen_b"}:
                        self._apply_pair_label("B", step=step)
                    elif cmd_l == "p":
                        self._apply_label("best")
                    elif cmd_l == "n":
                        self._apply_label("neutral")
                    elif cmd_l == "v":
                        _print_z("z_mean", self.state.z_mean)
                        _print_z("z_model", self.model.z_pref)
                        _print_z("z_a", z_a)
                        _print_z("z_b", z_b)
                    elif cmd_l in {"h", "history", "prev", "past"}:
                        path = self._review_history_menu()
                        if path is not None:
                            saved_path = str(path)
                            completed = True
                            raise StopIteration
                    elif cmd_l in {"a", "b"}:
                        choice = cmd_l.upper()
                        observed_action = "choice"
                        break
                    elif cmd_l == "f":
                        fb = self._feedback_menu()
                        if fb is not None:
                            feedback_label, feedback_strength = fb
                            observed_action = "feedback"
                            break
                    elif cmd_l == "s":
                        path = self.save_profile_menu(finished_after_soft_stop=self.ready_step is not None)
                        if path is not None:
                            saved_path = str(path)
                            completed = True
                            raise StopIteration
                        print("Сохранение отменено, продолжаем текущую пару.")
                    elif cmd_l == "q":
                        completed = False
                        raise StopIteration
                    else:
                        print("Неизвестная команда. Используй 1/2/a/b/f/p/n/v/h/s/q")
                        _print_session_commands()

                response_time = float(time.time() - pair_start)
                p_before = self.model.predict_proba_a(z_a, z_b)
                pred_before = "A" if p_before >= 0.5 else "B"

                if observed_action == "feedback" and feedback_label:
                    self.feedback_count += 1
                    fb_strength_mult, fb_std_decay, fb_model_mult, fb_phase = _feedback_phase_params(
                        self.session_cfg, step=step, ready_active=self.ready_step is not None
                    )
                    effective_strength = float(feedback_strength) * float(fb_strength_mult)
                    effective_model_lr = float(self.session_cfg.feedback_model_lr) * float(fb_model_mult)
                    self.state = apply_feedback_to_state(
                        state=self.state,
                        label=feedback_label,
                        strength=effective_strength,
                        clip_value=self.session_cfg.clip_value,
                        std_decay=fb_std_decay,
                        min_std=self.session_cfg.min_std,
                    )
                    feedback_model_record = apply_feedback_to_model(
                        model=self.model,
                        state_before=state_before,
                        state_after=self.state,
                        feedback_model_lr=effective_model_lr,
                    )
                    self.pending_feedback_recovery = {"feedback_label": feedback_label, "feedback_strength": effective_strength}
                    lr = 0.0
                    fb_phase_used = fb_phase
                else:
                    fb_phase_used = "none"
                    model_record = self.model.update(z_a, z_b, choice)
                    lr = self.session_cfg.semantic_lr if self.ready_step is None else self.session_cfg.semantic_lr * 0.70
                    self.state = update_state_from_choice(
                        state=self.state,
                        z_a=z_a,
                        z_b=z_b,
                        choice=choice,
                        lr=lr,
                        std_decay=self.session_cfg.std_decay,
                        min_std=self.session_cfg.min_std,
                        clip_value=self.session_cfg.clip_value,
                        pair_meta=pair_meta,
                    )

                update_norm = float(np.linalg.norm(self.state.z_mean - state_before.z_mean))
                self.recent_update_norms.append(update_norm)
                if self.ready_step is None and _is_soft_stop_candidate(step, self.session_cfg, self.state, update_norm, self.recent_update_norms):
                    self.ready_step = int(step)
                    ready_marker_set = True
                else:
                    ready_marker_set = False

                state_curve = self.mapper.map_one(self.state.z_mean)
                state_curve_metrics = mapped_curve_metrics(self.state.z_mean, self.mapper)
                pair_metrics = mapped_pair_metrics(z_a, z_b, self.mapper)
                step_payload = {
                    "step": int(step),
                    "strategy": self.strategy.name,
                    "pair_source": pair_source,
                    "pair_source_group": pair_meta.get("source_group"),
                    "contract_mode": pair_meta.get("contract_mode"),
                    "observed_action": observed_action,
                    "choice": choice,
                    "feedback_used": observed_action == "feedback",
                    "feedback_label": feedback_label,
                    "feedback_label_ru": _feedback_label_text(feedback_label) if feedback_label else None,
                    "feedback_strength": feedback_strength,
                    "feedback_phase": fb_phase_used,
                    "feedback_model_delta_norm": float(feedback_model_record.get("model_delta_norm", 0.0)),
                    "response_time_sec": response_time,
                    "p_before": float(p_before),
                    "pred_before": pred_before,
                    "loss_before": float(model_record.get("loss_before", np.nan)),
                    "loss_after": float(model_record.get("loss_after", np.nan)),
                    "applied_lr": float(lr),
                    "update_norm": update_norm,
                    "z_mean_norm": vector_norm(self.state.z_mean),
                    "z_model_norm": vector_norm(self.model.z_pref),
                    "mean_z_std": float(np.mean(self.state.z_std)),
                    "ready_marker_set": ready_marker_set,
                    "ready_step": self.ready_step,
                    "feedback_count_so_far": self.feedback_count,
                    **pair_metrics,
                    **state_curve_metrics,
                    "safety_ok": bool(pair_meta.get("safety_ok", True)),
                    "safety_shrink": float(pair_meta.get("safety_shrink", 1.0)),
                    "z_a": z_a,
                    "z_b": z_b,
                    "z_mean_before": state_before.z_mean,
                    "z_mean_after": self.state.z_mean,
                    "z_std_after": self.state.z_std,
                    "z_model_after": self.model.z_pref,
                    "eq_a_db": curve_a,
                    "eq_b_db": curve_b,
                    "eq_state_db": state_curve,
                }
                self.logger.log_step(step_payload)
                if observed_action == "choice" and choice in {"A", "B"}:
                    self._append_choice_history(
                        step=step,
                        choice=choice,
                        pair_source=pair_source,
                        pair_meta=pair_meta,
                        z_a=z_a,
                        z_b=z_b,
                        curve_a=curve_a,
                        curve_b=curve_b,
                        response_time=response_time,
                    )
                vec_row = {"step": step, "observed_action": observed_action, "choice": choice, "feedback_label": feedback_label}
                vec_row.update(flatten_vector("z_mean", FEATURE_NAMES_8D, self.state.z_mean))
                vec_row.update(flatten_vector("z_std", FEATURE_NAMES_8D, self.state.z_std))
                vec_row.update(flatten_vector("z_model", FEATURE_NAMES_8D, self.model.z_pref))
                vec_row.update(flatten_vector("z_a", FEATURE_NAMES_8D, z_a))
                vec_row.update(flatten_vector("z_b", FEATURE_NAMES_8D, z_b))
                self.logger.log_vectors(vec_row)
                self.logger.log_model_state({
                    "step": step,
                    "z_pref": self.model.z_pref,
                    "feature_weight": self.model.feature_weight,
                    "history_last": self.model.history[-1] if self.model.history else None,
                })

                if observed_action == "choice":
                    print(f"Выбор зафиксирован: {choice}. Обновил preference state, следующая пара построится вокруг новой области.")
                elif observed_action == "feedback":
                    print(f"Feedback применён: {_feedback_label_text(str(feedback_label))}. Следующая пара будет с учётом корректировки.")

                if ready_marker_set:
                    should_finish = self._handle_soft_stop(step)
                    if should_finish:
                        saved_path = self._aux_saved_profile_path or self._last_saved_profile_path or "saved_after_soft_stop"
                        completed = True
                        break

            if not completed:
                print("\nБюджет шагов закончился.")
                ans = _ask("Сохранить итоговый профиль? y/n", "y").lower()
                if ans.startswith("y"):
                    path = self.save_profile_interactive(finished_after_soft_stop=self.ready_step is not None)
                    saved_path = str(path)
                    completed = True
        except StopIteration:
            pass
        finally:
            try:
                self.audio.apply_neutral()
                audio_cfg = getattr(self.config, "audio", None)
                keep_alive = bool(getattr(audio_cfg, "keep_bridge_alive_between_sessions", True))
                if keep_alive:
                    self.logger.event("audio_bridge_kept_alive_between_sessions")
                    print("[audio] bridge kept alive for possible next session")
                else:
                    self.audio.stop()
            except Exception as exc:
                self.logger.event("audio_stop_error", error=str(exc))
            summary = {
                "session_id": self.logger.session_id,
                "strategy": self.strategy.name,
                "completed": completed,
                "saved_profile_path": saved_path,
                "steps_completed": int(self.state.step),
                "feedback_count": int(self.feedback_count),
                "soft_stop_triggered": self.ready_step is not None,
                "soft_stop_step": self.ready_step,
                "soft_stop_decisions": self.soft_stop_decisions,
                "choice_history_count": len(self.choice_history),
                "final_z_contract": self.state.z_mean,
                "final_z_std": self.state.z_std,
                "final_model_z_pref": self.model.z_pref,
                "final_state_z_contract": self.state.z_mean,
                "final_state_eq23_db": self.mapper.map_one(self.state.z_mean),
                "saved_profile_source": self._last_saved_profile_source,
                "saved_profile_source_step": self._last_saved_source_step,
                "saved_profile_source_choice": self._last_saved_source_choice,
                "saved_profile_z_contract": self._last_saved_z_contract,
                "saved_profile_eq23_db": self._last_saved_eq23_db,
                "last_selected_step": None if self._last_selected_entry() is None else self._last_selected_entry().step,
                "last_selected_choice": None if self._last_selected_entry() is None else self._last_selected_entry().choice,
                "last_selected_z_contract": None if self._last_selected_entry() is None else self._last_selected_entry().z_selected,
                "last_selected_eq23_db": None if self._last_selected_entry() is None else self._last_selected_entry().curve_selected,
                "audio_health": self.audio.health(),
                "mapper_version": self.mapper_version,
            }
            self.logger.save_summary(summary)
            self.logger.flush_tables()
            archive = None
            if self.config.logging.zip_on_finish:
                archive = self.logger.export_zip()
                print(f"\nLogs exported: {archive}")
            else:
                print(f"\nLogs folder: {self.logger.root}")
            self.logger.event("session_finished", completed=completed, archive=str(archive) if archive else None)

        return LiveSessionSummary(
            session_id=self.logger.session_id,
            strategy=self.strategy.name,
            completed=completed,
            saved_profile_path=saved_path,
            steps_completed=int(self.state.step),
            feedback_count=int(self.feedback_count),
            soft_stop_triggered=self.ready_step is not None,
            soft_stop_step=self.ready_step,
            final_rating=final_rating,
            final_z_contract=[float(x) for x in np.asarray(self.state.z_mean, dtype=float)],
        )
