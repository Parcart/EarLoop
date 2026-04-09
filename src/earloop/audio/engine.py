from __future__ import annotations

import queue
import threading
from typing import Any, Iterator, Optional

import numpy as np
import sounddevice as sd

from .processors import AudioProcessor, PassthroughProcessor

class AudioEngine:
    """
    Real-time bridge:
      capture device -> processor -> playback device

    Callback stays internal. The rest of the project sees a normal OO API.
    """

    def __init__(
        self,
        capture_device: int,
        playback_device: int,
        samplerate: int,
        channels: int = 2,
        blocksize: int = 1024,
        dtype: str = "float32",
        latency: str = "low",
        queue_blocks: int = 16,
        processor: Optional[AudioProcessor] = None,
        capture_extra_settings: Any | None = None,
    ):
        self.capture_device = capture_device
        self.playback_device = playback_device
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.dtype = dtype
        self.latency = latency
        self.capture_extra_settings = capture_extra_settings

        self.processor: AudioProcessor = processor or PassthroughProcessor()

        self._capture_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks)
        self._playback_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks)
        self._monitor_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks * 4)

        self._streams_lock = threading.RLock()
        self._worker_stop = threading.Event()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._input_stream = None
        self._output_stream = None

        self.xruns = {"input": 0, "output": 0, "worker": 0}

    def set_processor(self, processor: AudioProcessor) -> None:
        with self._streams_lock:
            self.processor = processor

    def get_debug_snapshot(self) -> dict[str, Any]:
        with self._streams_lock:
            input_stream = self._input_stream
            output_stream = self._output_stream
            processor = self.processor
            return {
                "running": self._running,
                "captureDevice": self.capture_device,
                "playbackDevice": self.playback_device,
                "samplerate": self.samplerate,
                "channels": self.channels,
                "blocksize": self.blocksize,
                "dtype": self.dtype,
                "latency": self.latency,
                "processorClass": processor.__class__.__name__ if processor is not None else None,
                "inputStreamActive": bool(getattr(input_stream, "active", False)) if input_stream is not None else False,
                "outputStreamActive": bool(getattr(output_stream, "active", False)) if output_stream is not None else False,
                "inputQueueSize": self._capture_queue.qsize(),
                "playbackQueueSize": self._playback_queue.qsize(),
                "monitorQueueSize": self._monitor_queue.qsize(),
                "xruns": dict(self.xruns),
            }

    def start(self) -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed. Install it before starting AudioEngine.")
        with self._streams_lock:
            if self._running:
                return
            self._worker_stop.clear()
            self._worker = threading.Thread(target=self._worker_loop, name="audio-worker", daemon=True)
            self._worker.start()

            self._input_stream = sd.InputStream(
                device=self.capture_device,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype=self.dtype,
                latency=self.latency,
                extra_settings=self.capture_extra_settings,
                callback=self._input_callback,
            )
            self._output_stream = sd.OutputStream(
                device=self.playback_device,
                samplerate=self.samplerate,
                blocksize=self.blocksize,
                channels=self.channels,
                dtype=self.dtype,
                latency=self.latency,
                callback=self._output_callback,
            )
            self._input_stream.start()
            self._output_stream.start()
            self._running = True

    def stop(self) -> None:
        with self._streams_lock:
            if not self._running:
                return
            self._worker_stop.set()

            for stream in (self._input_stream, self._output_stream):
                if stream is not None:
                    try:
                        stream.stop()
                    finally:
                        stream.close()

            self._input_stream = None
            self._output_stream = None
            self._running = False

        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

        self._drain_queue(self._capture_queue)
        self._drain_queue(self._playback_queue)

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "AudioEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    @staticmethod
    def list_devices() -> list[dict]:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.")
        return list(sd.query_devices())

    @staticmethod
    def print_devices() -> None:
        if sd is None:
            raise RuntimeError("sounddevice is not installed.")
        devices = sd.query_devices()
        hostapis = sd.query_hostapis()
        for i, dev in enumerate(devices):
            hostapi_name = hostapis[dev["hostapi"]]["name"]
            print(
                f"{i:>3} | {dev['name']} | hostapi={hostapi_name} | "
                f"in={dev['max_input_channels']} | out={dev['max_output_channels']} | "
                f"default_sr={int(dev['default_samplerate'])}"
            )

    def iter_monitor(self) -> Iterator[np.ndarray]:
        """Optional generator for GUI meters or debug consumers."""
        while self._running or not self._monitor_queue.empty():
            try:
                yield self._monitor_queue.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break

    def _input_callback(self, indata, frames, time, status) -> None:
        if status:
            self.xruns["input"] += 1
        chunk = np.array(indata, copy=True)
        self._put_drop_oldest(self._capture_queue, chunk, xrun_key="input")

    def _output_callback(self, outdata, frames, time, status) -> None:
        if status:
            self.xruns["output"] += 1
        try:
            block = self._playback_queue.get_nowait()
            if block.shape != outdata.shape:
                outdata.fill(0)
            else:
                outdata[:] = block
        except queue.Empty:
            outdata.fill(0)

    def _worker_loop(self) -> None:
        while not self._worker_stop.is_set():
            try:
                chunk = self._capture_queue.get(timeout=0.2)
            except queue.Empty:
                continue

            try:
                processed = self.processor.process(chunk)
                processed = np.asarray(processed, dtype=np.float32)
                self._put_drop_oldest(self._playback_queue, processed, xrun_key="worker")
                self._put_drop_oldest(self._monitor_queue, processed, xrun_key=None)
            except Exception:
                self.xruns["worker"] += 1

    def _put_drop_oldest(self, q: queue.Queue[np.ndarray], block: np.ndarray, xrun_key: Optional[str]) -> None:
        try:
            q.put_nowait(block)
        except queue.Full:
            try:
                q.get_nowait()
            except queue.Empty:
                pass
            try:
                q.put_nowait(block)
            except queue.Full:
                if xrun_key is not None:
                    self.xruns[xrun_key] += 1

    @staticmethod
    def _drain_queue(q: queue.Queue[np.ndarray]) -> None:
        while True:
            try:
                q.get_nowait()
            except queue.Empty:
                break
