import argparse
import queue
import sys
import sounddevice as sd
import numpy as np


def list_devices():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()
    print("Доступные аудиоустройства:\n")
    for i, dev in enumerate(devices):
        hostapi_name = hostapis[dev["hostapi"]]["name"]
        print(
            f"{i:>3} | {dev['name']} | hostapi={hostapi_name} | "
            f"in={dev['max_input_channels']} | out={dev['max_output_channels']} | "
            f"default_sr={int(dev['default_samplerate'])}"
        )


def find_input_device_by_name(substr_list):
    devices = sd.query_devices()
    for i, dev in enumerate(devices):
        name = dev["name"].lower()
        if dev["max_input_channels"] > 0 and any(s.lower() in name for s in substr_list):
            return i
    return None


def choose_output_device():
    devices = sd.query_devices()
    hostapis = sd.query_hostapis()

    print("\nУстройства вывода:\n")
    for i, dev in enumerate(devices):
        if dev["max_output_channels"] > 0:
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            print(
                f"{i:>3} | {dev['name']} | hostapi={hostapi_name} | "
                f"out={dev['max_output_channels']} | default_sr={int(dev['default_samplerate'])}"
            )

    while True:
        raw = input("\nВведите индекс устройства вывода: ").strip()
        try:
            idx = int(raw)
            info = sd.query_devices(idx)
            if info["max_output_channels"] > 0:
                return idx
            print("Это не устройство вывода.")
        except Exception:
            print("Неверный индекс, попробуй снова.")


def pick_working_format(input_dev, output_dev, user_samplerate=None, user_channels=None, dtype="float32"):
    in_info = sd.query_devices(input_dev)
    out_info = sd.query_devices(output_dev)

    channel_candidates = []
    if user_channels is not None:
        channel_candidates.append(int(user_channels))

    max_common_channels = int(min(in_info["max_input_channels"], out_info["max_output_channels"]))
    if max_common_channels >= 2:
        channel_candidates.extend([2, 1])
    elif max_common_channels >= 1:
        channel_candidates.append(1)

    if max_common_channels > 0 and max_common_channels not in channel_candidates:
        channel_candidates.append(max_common_channels)

    sr_candidates = []
    if user_samplerate is not None:
        sr_candidates.append(int(user_samplerate))

    for sr in [
        int(round(in_info["default_samplerate"])),
        int(round(out_info["default_samplerate"])),
        48000,
        44100,
    ]:
        if sr not in sr_candidates:
            sr_candidates.append(sr)

    for channels in channel_candidates:
        for sr in sr_candidates:
            try:
                sd.check_input_settings(device=input_dev, channels=channels, samplerate=sr, dtype=dtype)
                sd.check_output_settings(device=output_dev, channels=channels, samplerate=sr, dtype=dtype)
                return sr, channels
            except Exception:
                continue

    raise RuntimeError(
        "Не удалось подобрать общие samplerate/channels. "
        "Попробуй явно: --samplerate 48000 --channels 2"
    )


def process_audio(indata: np.ndarray) -> np.ndarray:
    # Здесь твоя обработка
    return indata


def main():
    parser = argparse.ArgumentParser(description="VB-CABLE -> обработка -> выбранный output")
    parser.add_argument("--list", action="store_true", help="Показать список устройств и выйти")
    parser.add_argument("--input-device", type=int, default=None, help="Индекс input-устройства (по умолчанию ищется CABLE Output)")
    parser.add_argument("--output-device", type=int, default=None, help="Индекс output-устройства")
    parser.add_argument("--samplerate", type=int, default=None, help="Например 48000")
    parser.add_argument("--channels", type=int, default=None, help="1 или 2")
    parser.add_argument("--blocksize", type=int, default=1024, help="Размер блока")
    parser.add_argument("--dtype", type=str, default="float32", choices=["float32", "int16", "int32"])
    parser.add_argument("--latency", type=str, default="low", choices=["low", "high"])
    parser.add_argument("--queue-blocks", type=int, default=16, help="Размер очереди в блоках")
    args = parser.parse_args()

    if args.list:
        list_devices()
        return

    if args.input_device is None:
        input_device = find_input_device_by_name([
            "CABLE Output",
            "VB-Audio Point",
            "VB-Audio Virtual Cable",
        ])
        if input_device is None:
            print("Не удалось автоматически найти CABLE Output.")
            print("Сначала запусти: python script.py --list")
            print("Потом укажи вручную: --input-device N")
            sys.exit(1)
    else:
        input_device = args.input_device

    if args.output_device is None:
        output_device = choose_output_device()
    else:
        output_device = args.output_device

    if input_device == output_device:
        print("Input и output не должны быть одним и тем же устройством.")
        sys.exit(1)

    try:
        samplerate, channels = pick_working_format(
            input_dev=input_device,
            output_dev=output_device,
            user_samplerate=args.samplerate,
            user_channels=args.channels,
            dtype=args.dtype,
        )
    except Exception as e:
        print(f"Ошибка подбора формата: {e}")
        sys.exit(1)

    in_info = sd.query_devices(input_device)
    out_info = sd.query_devices(output_device)
    hostapis = sd.query_hostapis()
    in_hostapi = hostapis[in_info["hostapi"]]["name"]
    out_hostapi = hostapis[out_info["hostapi"]]["name"]

    print("\nЗапуск аудиомоста:")
    print(f"  INPUT : {input_device} | {in_info['name']} | hostapi={in_hostapi}")
    print(f"  OUTPUT: {output_device} | {out_info['name']} | hostapi={out_hostapi}")
    print(f"  SR={samplerate}, channels={channels}, blocksize={args.blocksize}, dtype={args.dtype}, latency={args.latency}")
    print("\nНажми Ctrl+C для остановки.\n")

    q = queue.Queue(maxsize=args.queue_blocks)

    def input_callback(indata, frames, time, status):
        if status:
            print(f"[input status] {status}", file=sys.stderr)

        try:
            processed = process_audio(indata.copy())
            if processed.shape != indata.shape:
                processed = indata.copy()

            try:
                q.put_nowait(processed)
            except queue.Full:
                try:
                    q.get_nowait()  # выкидываем самый старый блок
                except queue.Empty:
                    pass
                try:
                    q.put_nowait(processed)
                except queue.Full:
                    pass
        except Exception as e:
            print(f"[input callback error] {e}", file=sys.stderr)

    def output_callback(outdata, frames, time, status):
        if status:
            print(f"[output status] {status}", file=sys.stderr)

        try:
            block = q.get_nowait()
            if block.shape != outdata.shape:
                outdata.fill(0)
            else:
                outdata[:] = block
        except queue.Empty:
            outdata.fill(0)

    try:
        with sd.InputStream(
            device=input_device,
            samplerate=samplerate,
            blocksize=args.blocksize,
            channels=channels,
            dtype=args.dtype,
            latency=args.latency,
            callback=input_callback,
        ), sd.OutputStream(
            device=output_device,
            samplerate=samplerate,
            blocksize=args.blocksize,
            channels=channels,
            dtype=args.dtype,
            latency=args.latency,
            callback=output_callback,
        ):
            while True:
                sd.sleep(1000)

    except KeyboardInterrupt:
        print("\nОстановлено пользователем.")
    except Exception as e:
        print(f"\nОшибка запуска: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()