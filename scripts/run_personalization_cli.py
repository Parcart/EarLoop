from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

try:
    import sounddevice as sd
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "Не удалось импортировать sounddevice. Установи зависимости: pip install sounddevice numpy scipy"
    ) from exc

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_DIR = PROJECT_ROOT / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from earloop.audio import AudioEngine, EqProfile, EqualizerProcessor  # noqa: E402
from earloop.ml.generation.pool import PoolCandidateGenerator  # noqa: E402
from earloop.ml.mapping import ParametricEqMapper  # noqa: E402
from earloop.ml.pipeline import PersonalizationPipeline  # noqa: E402
from earloop.ml.preference.linear import LinearPreferenceModel  # noqa: E402
from earloop.ml.types import PerceptualProfile, PreferenceChoice  # noqa: E402
from earloop.utils.logging_utils import append_jsonl, setup_logger # noqa: E402

DEFAULT_FEATURES = ["bass", "tilt", "presence", "air", "lowmid", "sparkle"]
DEFAULT_EQ_FREQS = np.array(
    [25, 31.5, 40, 50, 63, 80, 100, 125, 160, 200, 250, 315,
     400, 500, 630, 800, 1000, 1600, 2500, 4000, 6300, 10000, 16000],
    dtype=np.float32,
)


def parse_csv_strings(value: str) -> list[str]:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Список не должен быть пустым")
    return parts


def hostapi_name(dev: dict) -> str:
    hostapis = sd.query_hostapis()
    return hostapis[dev["hostapi"]]["name"]


def print_devices() -> None:
    print("\nДоступные аудиоустройства:\n")
    for idx, dev in enumerate(sd.query_devices()):
        print(
            f"{idx:>3} | {dev['name']} | hostapi={hostapi_name(dev)} | "
            f"in={dev['max_input_channels']} | out={dev['max_output_channels']} | "
            f"default_sr={int(dev['default_samplerate'])}"
        )


def iter_matching_input_devices(name_parts: Iterable[str]) -> Iterable[int]:
    lowered = [p.lower() for p in name_parts]
    for idx, dev in enumerate(sd.query_devices()):
        if dev["max_input_channels"] <= 0:
            continue
        name = dev["name"].lower()
        if any(part in name for part in lowered):
            yield idx


def choose_device(kind: str, require_input: bool = False, require_output: bool = False, default: int | None = None) -> int:
    assert require_input or require_output
    while True:
        prompt = f"Введите индекс устройства {kind}"
        if default is not None:
            prompt += f" [Enter = {default}]"
        prompt += ": "
        raw = input(prompt).strip()

        if raw == "" and default is not None:
            idx = default
        else:
            try:
                idx = int(raw)
            except ValueError:
                print("Нужно ввести целое число.")
                continue

        try:
            dev = sd.query_devices(idx)
        except Exception:
            print("Устройства с таким индексом нет.")
            continue

        if require_input and dev["max_input_channels"] <= 0:
            print("Это не input-устройство.")
            continue
        if require_output and dev["max_output_channels"] <= 0:
            print("Это не output-устройство.")
            continue
        return idx


def pick_working_format(input_device: int, output_device: int, samplerate: int | None, channels: int | None, dtype: str) -> tuple[int, int]:
    in_info = sd.query_devices(input_device)
    out_info = sd.query_devices(output_device)

    channel_candidates: list[int] = []
    if channels is not None:
        channel_candidates.append(int(channels))

    common_channels = int(min(in_info["max_input_channels"], out_info["max_output_channels"]))
    if common_channels >= 2:
        channel_candidates.extend([2, 1])
    elif common_channels >= 1:
        channel_candidates.append(1)

    if common_channels > 0 and common_channels not in channel_candidates:
        channel_candidates.append(common_channels)

    sr_candidates: list[int] = []
    if samplerate is not None:
        sr_candidates.append(int(samplerate))
    for sr in [
        int(round(in_info["default_samplerate"])),
        int(round(out_info["default_samplerate"])),
        48000,
        44100,
    ]:
        if sr not in sr_candidates:
            sr_candidates.append(sr)

    for ch in channel_candidates:
        for sr in sr_candidates:
            try:
                sd.check_input_settings(device=input_device, channels=ch, samplerate=sr, dtype=dtype)
                sd.check_output_settings(device=output_device, channels=ch, samplerate=sr, dtype=dtype)
                return sr, ch
            except Exception:
                continue

    raise RuntimeError(
        "Не удалось подобрать общие samplerate/channels для выбранных устройств. "
        "Попробуй явно указать --samplerate 48000 --channels 2"
    )


def read_profiles_csv(path: str | Path, feature_names: Sequence[str]) -> list[PerceptualProfile]:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Файл профилей не найден: {path}")

    profiles: list[PerceptualProfile] = []
    with path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        missing = [name for name in feature_names if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError(f"В CSV нет колонок: {missing}")

        for row_idx, row in enumerate(reader):
            values = np.array([float(row[name]) for name in feature_names], dtype=np.float32)
            profile_id = row.get("profile_id") or row.get("id") or str(row_idx)
            label = row.get("label") or row.get("name")
            profiles.append(
                PerceptualProfile(
                    values=values,
                    profile_id=str(profile_id),
                    label=label,
                    metadata={"row_index": row_idx},
                )
            )

    if len(profiles) < 2:
        raise ValueError("Для генератора нужно минимум 2 профиля")
    return profiles


def generate_random_profiles(count: int, feature_names: Sequence[str], seed: int | None) -> list[PerceptualProfile]:
    if count < 2:
        raise ValueError("count must be >= 2")
    rng = np.random.default_rng(seed)
    profiles: list[PerceptualProfile] = []
    for idx in range(count):
        values = rng.uniform(-1.0, 1.0, size=len(feature_names)).astype(np.float32)
        profiles.append(
            PerceptualProfile(
                values=values,
                profile_id=f"rand_{idx}",
                label=f"rand_{idx}",
            )
        )
    return profiles


def to_audio_eq_profile(pair_side_name: str, pair_id: str | None, eq_curve) -> EqProfile:
    profile_name = f"{pair_id or 'pair'}_{pair_side_name}"
    return EqProfile(
        profile_id=0,
        freqs_hz=np.asarray(eq_curve.freqs_hz, dtype=np.float64),
        gains_db=np.asarray(eq_curve.gains_db, dtype=np.float64),
        preamp_db=float(eq_curve.preamp_db),
        name=profile_name,
    )


def print_pair_summary(candidate_pair) -> None:
    left = candidate_pair.left
    right = candidate_pair.right
    print("\n" + "=" * 72)
    print(f"Итерация: {candidate_pair.iteration} | pair_id={candidate_pair.pair_id}")
    print("LEFT")
    print(f"  perceptual: {np.round(left.perceptual.values, 3).tolist()}")
    print(f"  score: {left.score:.4f}" if left.score is not None else "  score: n/a")
    print("RIGHT")
    print(f"  perceptual: {np.round(right.perceptual.values, 3).tolist()}")
    print(f"  score: {right.score:.4f}" if right.score is not None else "  score: n/a")
    print("Команды:")
    print("  a  - включить LEFT")
    print("  b  - включить RIGHT")
    print("  1  - выбрать LEFT")
    print("  2  - выбрать RIGHT")
    print("  n  - ни один не подходит (REJECT_BOTH)")
    print("  s  - пропустить пару (SKIP)")
    print("  p  - показать текущие веса модели")
    print("  x  - показать xruns аудиодвижка")
    print("  q  - завершить сессию")
    print("=" * 72)


def prompt_choice() -> str:
    allowed = {"a", "b", "1", "2", "n", "s", "p", "x", "q"}
    while True:
        cmd = input("Введите команду: ").strip().lower()
        if cmd in allowed:
            return cmd
        print("Неизвестная команда.")


def save_json(path: str | Path, data: object) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def dataclass_or_obj_to_dict(obj: object) -> object:
    if is_dataclass(obj):
        return asdict(obj)
    if isinstance(obj, dict):
        return {k: dataclass_or_obj_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [dataclass_or_obj_to_dict(v) for v in obj]
    return obj


def load_model_state_if_exists(model: LinearPreferenceModel, path: str | Path | None) -> bool:
    if path is None:
        return False
    path = Path(path)
    if not path.exists():
        return False

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    # Поддержка простого JSON-состояния вида get_state()->asdict(...)
    weights = np.asarray(data.get("weights", []), dtype=np.float32)
    dim = int(data.get("dim", model.dim))
    if dim != model.dim:
        raise ValueError(f"Размерность в state ({dim}) не совпадает с model.dim ({model.dim})")

    model.learning_rate = float(data.get("learning_rate", model.learning_rate))
    model.l2_reg = float(data.get("l2_reg", model.l2_reg))
    model.set_weights(weights)
    return True


def main() -> None:
    parser = argparse.ArgumentParser(description="Консольная A/B-сессия персонализации эквалайзера")
    parser.add_argument("--list", action="store_true", help="Показать список аудиоустройств и выйти")
    parser.add_argument("--profiles-csv", type=str, default=None, help="CSV с perceptual-профилями")
    parser.add_argument("--feature-names", type=parse_csv_strings, default=DEFAULT_FEATURES, help="Список perceptual-параметров")
    parser.add_argument("--random-pool-size", type=int, default=64, help="Если CSV не задан, создать случайный пул такого размера")
    parser.add_argument("--input-device", type=int, default=None)
    parser.add_argument("--output-device", type=int, default=None)
    parser.add_argument("--samplerate", type=int, default=None)
    parser.add_argument("--channels", type=int, default=None)
    parser.add_argument("--blocksize", type=int, default=1024)
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "int16", "int32"])
    parser.add_argument("--latency", type=str, default="low", choices=["low", "high"])
    parser.add_argument("--eq-freqs", type=lambda s: np.array([float(x.strip()) for x in s.split(",") if x.strip()], dtype=np.float32), default=DEFAULT_EQ_FREQS)
    parser.add_argument("--explore-prob", type=float, default=0.25)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--learning-rate", type=float, default=0.10)
    parser.add_argument("--l2-reg", type=float, default=0.00)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--user-id", type=str, default="local_user")
    parser.add_argument("--session-id", type=str, default=None)
    parser.add_argument("--state-path", type=str, default=None, help="JSON-файл со state модели для загрузки/сохранения")
    parser.add_argument("--events-path", type=str, default=None, help="JSON-файл для накопленного event-log")
    parser.add_argument(
        "--events-jsonl",
        type=str,
        default="data/logs/personalization_events.jsonl",
        help="JSONL-лог событий сессии",
    )
    args = parser.parse_args()

    logger = setup_logger("personalization-cli")

    logger.info("CLI arguments parsed")

    if args.list:
        print_devices()
        return

    print_devices()

    if args.input_device is None:
        default_input = next(iter_matching_input_devices(["CABLE Output", "VB-Audio", "Virtual Cable"]), None)
        if default_input is not None:
            print(f"\nАвтоматически найден вероятный источник захвата: {default_input}")
        input_device = choose_device("ввода", require_input=True, default=default_input)
    else:
        input_device = args.input_device

    if args.output_device is None:
        output_device = choose_device("вывода", require_output=True)
    else:
        output_device = args.output_device

    if input_device == output_device:
        raise SystemExit("Input и output не должны быть одним и тем же устройством.")

    samplerate, channels = pick_working_format(
        input_device=input_device,
        output_device=output_device,
        samplerate=args.samplerate,
        channels=args.channels,
        dtype=args.dtype,
    )

    if args.profiles_csv is not None:
        profiles = read_profiles_csv(args.profiles_csv, args.feature_names)
        source_info = f"CSV: {args.profiles_csv}"
    else:
        profiles = generate_random_profiles(args.random_pool_size, args.feature_names, seed=args.seed)
        source_info = f"random pool size={args.random_pool_size}"

    model = LinearPreferenceModel(
        dim=len(args.feature_names),
        learning_rate=args.learning_rate,
        l2_reg=args.l2_reg,
    )
    state_loaded = load_model_state_if_exists(model, args.state_path)

    generator = PoolCandidateGenerator(
        profiles=profiles,
        explore_prob=args.explore_prob,
        top_k=args.top_k,
        random_seed=args.seed,
    )
    mapper = ParametricEqMapper(freqs_hz=args.eq_freqs, feature_names=args.feature_names)

    session_id = args.session_id or f"session_{int(time.time())}"
    pipeline = PersonalizationPipeline(
        preference_model=model,
        candidate_generator=generator,
        eq_mapper=mapper,
        eq_validator=None,
        user_id=args.user_id,
        session_id=session_id,
        model_version="linear_v1",
        mapper_version="parametric_eq_v1",
    )

    neutral_curve = mapper.map_profile(PerceptualProfile(values=np.zeros(len(args.feature_names), dtype=np.float32)))
    processor = EqualizerProcessor(
        samplerate=samplerate,
        channels=channels,
        initial_profile=to_audio_eq_profile("neutral", "init", neutral_curve),
    )
    engine = AudioEngine(
        capture_device=input_device,
        playback_device=output_device,
        samplerate=samplerate,
        channels=channels,
        blocksize=args.blocksize,
        processor=processor,
        dtype=args.dtype,
        latency=args.latency,
    )

    in_dev = sd.query_devices(input_device)
    out_dev = sd.query_devices(output_device)

    logger.info(
        f"Session start | user_id={args.user_id} | session_id={session_id} | "
        f"profiles={len(profiles)} | features={args.feature_names}"
    )

    append_jsonl(
        args.events_jsonl,
        "session_started",
        user_id=args.user_id,
        session_id=session_id,
        input_device=input_device,
        output_device=output_device,
        samplerate=samplerate,
        channels=channels,
        feature_names=list(args.feature_names),
        profile_source=source_info,
        state_loaded=state_loaded,
    )

    print("\nЗапуск personalization CLI:")
    print(f"  INPUT : {input_device} | {in_dev['name']} | hostapi={hostapi_name(in_dev)}")
    print(f"  OUTPUT: {output_device} | {out_dev['name']} | hostapi={hostapi_name(out_dev)}")
    print(f"  SR={samplerate}, channels={channels}, blocksize={args.blocksize}, dtype={args.dtype}, latency={args.latency}")
    print(f"  feature_names={args.feature_names}")
    print(f"  profile source={source_info}")
    print(f"  model state loaded={state_loaded}")
    print("\nСхема работы:")
    print("  Windows/приложение -> CABLE Input -> CABLE Output -> Python EQ -> выбранное устройство")
    print("\nВажно: в этой сессии A/B-пара выбирается в консоли, а активный EQ применяется к живому потоку.")
    print("Нажми Ctrl+C или q для остановки.\n")

    engine.start()

    previous_left_profile_id = None
    try:
        while True:
            candidate_pair = pipeline.next_candidate_pair()

            left_profile_id = candidate_pair.left.perceptual.profile_id
            right_profile_id = candidate_pair.right.perceptual.profile_id
            left_repeated = previous_left_profile_id == left_profile_id
            previous_left_profile_id = left_profile_id

            logger.info(
                f"Pair generated | iter={candidate_pair.iteration} | "
                f"left={left_profile_id} | right={right_profile_id} | "
                f"left_score={candidate_pair.left.score:.4f} | right_score={candidate_pair.right.score:.4f} | "
                f"left_repeated={left_repeated}"
            )

            append_jsonl(
                args.events_jsonl,
                "pair_generated",
                user_id=args.user_id,
                session_id=session_id,
                pair_id=candidate_pair.pair_id,
                iteration=candidate_pair.iteration,
                left_profile_id=left_profile_id,
                right_profile_id=right_profile_id,
                left_values=candidate_pair.left.perceptual.values.tolist(),
                right_values=candidate_pair.right.perceptual.values.tolist(),
                left_score=float(candidate_pair.left.score) if candidate_pair.left.score is not None else None,
                right_score=float(candidate_pair.right.score) if candidate_pair.right.score is not None else None,
                left_repeated=left_repeated,
            )

            left_audio_profile = to_audio_eq_profile("left", candidate_pair.pair_id, candidate_pair.left.eq_curve)
            right_audio_profile = to_audio_eq_profile("right", candidate_pair.pair_id, candidate_pair.right.eq_curve)

            processor.set_profile(left_audio_profile)
            active_side = "LEFT"
            print_pair_summary(candidate_pair)
            print("Сейчас активен: LEFT")

            previous_left_profile_id = None

            while True:
                cmd = prompt_choice()

                if cmd == "a":
                    processor.set_profile(left_audio_profile)
                    active_side = "LEFT"
                    print("Активен LEFT")
                    logger.info(f"Preview switch | pair_id={candidate_pair.pair_id} | active=LEFT")
                    append_jsonl(
                        args.events_jsonl,
                        "preview_switch",
                        user_id=args.user_id,
                        session_id=session_id,
                        pair_id=candidate_pair.pair_id,
                        iteration=candidate_pair.iteration,
                        active_side="LEFT",
                    )
                    continue
                if cmd == "b":
                    processor.set_profile(right_audio_profile)
                    active_side = "RIGHT"
                    print("Активен RIGHT")
                    logger.info(f"Preview switch | pair_id={candidate_pair.pair_id} | active=RIGHT")
                    append_jsonl(
                        args.events_jsonl,
                        "preview_switch",
                        user_id=args.user_id,
                        session_id=session_id,
                        pair_id=candidate_pair.pair_id,
                        iteration=candidate_pair.iteration,
                        active_side="RIGHT",
                    )
                    continue
                if cmd == "p":
                    print("weights:", np.round(model.weights, 4).tolist())
                    print("active side:", active_side)
                    continue
                if cmd == "x":
                    print("xruns:", engine.xruns)
                    continue
                if cmd == "q":
                    pipeline.submit_choice(PreferenceChoice.STOP)
                    raise KeyboardInterrupt
                if cmd == "s":
                    pipeline.submit_choice(PreferenceChoice.SKIP)
                    break
                if cmd == "n":
                    pipeline.submit_choice(PreferenceChoice.REJECT_BOTH)
                    break
                if cmd == "1":
                    pipeline.submit_choice(PreferenceChoice.LEFT)
                    print("Выбран LEFT")
                    logger.info(f"User choice | pair_id={candidate_pair.pair_id} | choice=LEFT")
                    append_jsonl(
                        args.events_jsonl,
                        "user_choice",
                        user_id=args.user_id,
                        session_id=session_id,
                        pair_id=candidate_pair.pair_id,
                        iteration=candidate_pair.iteration,
                        choice="LEFT",
                        active_side=active_side,
                        left_profile_id=left_profile_id,
                        right_profile_id=right_profile_id,
                    )
                    break
                if cmd == "2":
                    pipeline.submit_choice(PreferenceChoice.RIGHT)
                    print("Выбран RIGHT")
                    logger.info(f"User choice | pair_id={candidate_pair.pair_id} | choice=RIGHT")
                    append_jsonl(
                        args.events_jsonl,
                        "user_choice",
                        user_id=args.user_id,
                        session_id=session_id,
                        pair_id=candidate_pair.pair_id,
                        iteration=candidate_pair.iteration,
                        choice="RIGHT",
                        active_side=active_side,
                        left_profile_id=left_profile_id,
                        right_profile_id=right_profile_id,
                    )
                    break

            if args.state_path:
                state = model.get_state()
                save_json(args.state_path, dataclass_or_obj_to_dict(state))
            if args.events_path:
                save_json(args.events_path, dataclass_or_obj_to_dict(pipeline.event_log))

    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        engine.stop()

        if args.state_path:
            state = model.get_state()
            save_json(args.state_path, dataclass_or_obj_to_dict(state))
            print(f"Состояние модели сохранено: {args.state_path}")
        if args.events_path:
            save_json(args.events_path, dataclass_or_obj_to_dict(pipeline.event_log))
            print(f"Логи выбора сохранены: {args.events_path}")

        print("Готово.")

        logger.info("Session finished")

        append_jsonl(
            args.events_jsonl,
            "session_finished",
            user_id=args.user_id,
            session_id=session_id,
            iterations=pipeline.iteration,
            event_count=len(pipeline.event_log),
            xruns=engine.xruns,
        )


if __name__ == "__main__":
    main()
