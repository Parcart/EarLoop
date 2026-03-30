from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Iterable, Optional

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


DEFAULT_FREQS = np.array([31.25, 62.5, 125.0, 250.0, 500.0, 1000.0, 2000.0, 4000.0, 8000.0, 16000.0], dtype=np.float64)
DEFAULT_GAINS = np.array([0.0, 0.0, 1.5, 2.0, 0.5, -1.0, 1.0, 2.5, 1.5, 0.0], dtype=np.float64)


def parse_csv_floats(value: str) -> np.ndarray:
    parts = [p.strip() for p in value.split(",") if p.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("Список не должен быть пустым")
    try:
        return np.array([float(p) for p in parts], dtype=np.float64)
    except ValueError as exc:
        raise argparse.ArgumentTypeError("Не удалось разобрать список чисел") from exc


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


def choose_device(kind: str, require_input: bool = False, require_output: bool = False, default: Optional[int] = None) -> int:
    assert require_input or require_output

    while True:
        if default is not None:
            prompt = f"Введите индекс устройства {kind} [Enter = {default}]: "
        else:
            prompt = f"Введите индекс устройства {kind}: "
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


def pick_working_format(input_device: int, output_device: int, samplerate: Optional[int], channels: Optional[int], dtype: str) -> tuple[int, int]:
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


def build_profile(freqs: np.ndarray, gains: np.ndarray, preamp_db: float, name: str) -> EqProfile:
    if freqs.ndim != 1 or gains.ndim != 1:
        raise ValueError("freqs и gains должны быть одномерными")
    if len(freqs) != len(gains):
        raise ValueError("Количество частот и коэффициентов усиления должно совпадать")
    if len(freqs) == 0:
        raise ValueError("Профиль EQ не должен быть пустым")
    return EqProfile(
        profile_id=1,
        freqs_hz=np.asarray(freqs, dtype=np.float64),
        gains_db=np.asarray(gains, dtype=np.float64),
        preamp_db=float(preamp_db),
        name=name,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Запуск live-аудиомоста с эквалайзером")
    parser.add_argument("--list", action="store_true", help="Показать список устройств и выйти")
    parser.add_argument("--input-device", type=int, default=None, help="Индекс input-устройства захвата")
    parser.add_argument("--output-device", type=int, default=None, help="Индекс output-устройства воспроизведения")
    parser.add_argument("--samplerate", type=int, default=None, help="Например 48000")
    parser.add_argument("--channels", type=int, default=None, help="Например 1 или 2")
    parser.add_argument("--blocksize", type=int, default=1024, help="Размер блока аудио")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "int16", "int32"])
    parser.add_argument("--latency", type=str, default="low", choices=["low", "high"])
    parser.add_argument("--freqs", type=parse_csv_floats, default=DEFAULT_FREQS, help="Частоты EQ через запятую")
    parser.add_argument("--gains", type=parse_csv_floats, default=DEFAULT_GAINS, help="Усиления EQ в dB через запятую")
    parser.add_argument("--preamp-db", type=float, default=-3.0, help="Предусиление в dB")
    parser.add_argument("--profile-name", type=str, default="live-eq", help="Имя профиля")
    args = parser.parse_args()

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

    profile = build_profile(args.freqs, args.gains, args.preamp_db, args.profile_name)
    processor = EqualizerProcessor(samplerate=samplerate, channels=channels, initial_profile=profile)

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

    print("\nЗапуск live-аудиомоста:")
    print(f"  INPUT : {input_device} | {in_dev['name']} | hostapi={hostapi_name(in_dev)}")
    print(f"  OUTPUT: {output_device} | {out_dev['name']} | hostapi={hostapi_name(out_dev)}")
    print(f"  SR={samplerate}, channels={channels}, blocksize={args.blocksize}, dtype={args.dtype}, latency={args.latency}")
    print(f"  EQ profile: {profile.name}")
    print(f"  freqs={profile.freqs_hz.tolist()}")
    print(f"  gains={profile.gains_db.tolist()} dB, preamp={profile.preamp_db} dB")
    print("\nСхема работы:")
    print("  Windows/приложение -> CABLE Input -> CABLE Output -> Python EQ -> выбранное устройство")
    print("\nНажми Ctrl+C для остановки.\n")

    engine.start()
    try:
        while True:
            time.sleep(2.0)
            xr = engine.xruns
            print(
                f"status | input_xruns={xr['input']} | output_xruns={xr['output']} | worker_errors={xr['worker']}",
                end="\r",
                flush=True,
            )
    except KeyboardInterrupt:
        print("\nОстановка...")
    finally:
        engine.stop()
        print("Готово.")


if __name__ == "__main__":
    main()
