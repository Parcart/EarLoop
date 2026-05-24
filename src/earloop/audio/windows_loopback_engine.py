from __future__ import annotations

import ctypes
import queue
import threading
import traceback
from importlib.util import find_spec
from typing import Any, Iterator, Optional

import numpy as np
import soundcard as sc
import sounddevice as sd

from .processors import AudioProcessor, PassthroughProcessor
from earloop.utils.logging_utils import setup_logger


INITIAL_DIAGNOSTIC_CHUNKS = 32
SILENCE_RMS_THRESHOLD = 1.0e-4
SILENCE_PEAK_THRESHOLD = 5.0e-4
PERSISTENT_SILENCE_LOG_EVERY = 50
MAX_CAPTURE_ANOMALY_LOGS = 4


class WindowsLoopbackAudioEngine:
    """
    Windows-only real-time bridge:
      WASAPI render loopback capture -> processor -> playback device

    Capture is done via `soundcard` to avoid the current PortAudio loopback
    limitation. Processing and playback stay aligned with the existing
    EarLoop pipeline expectations.
    """

    def __init__(
        self,
        loopback_endpoint_id: str,
        playback_device: int,
        samplerate: int,
        channels: int = 2,
        blocksize: int = 1024,
        dtype: str = "float32",
        latency: str = "low",
        queue_blocks: int = 16,
        processor: Optional[AudioProcessor] = None,
    ):
        self._logger = setup_logger("earloop.windows-loopback")
        self.loopback_endpoint_id = loopback_endpoint_id
        self.playback_device = playback_device
        self.samplerate = int(samplerate)
        self.channels = int(channels)
        self.blocksize = int(blocksize)
        self.dtype = dtype
        self.latency = latency

        self.processor: AudioProcessor = processor or PassthroughProcessor()

        self._capture_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks)
        self._playback_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks)
        self._monitor_queue: queue.Queue[np.ndarray] = queue.Queue(maxsize=queue_blocks * 4)

        self._streams_lock = threading.RLock()
        self._worker_stop = threading.Event()
        self._capture_stop = threading.Event()
        self._running = False
        self._worker: Optional[threading.Thread] = None
        self._capture_worker: Optional[threading.Thread] = None
        self._output_stream = None
        self._capture_ready = threading.Event()
        self._capture_error: Optional[BaseException] = None

        self.xruns = {"input": 0, "output": 0, "worker": 0}
        self._diagnostic_chunk_index = 0
        self._consecutive_silent_chunks = 0
        self._capture_anomaly_logs = 0

    def set_processor(self, processor: AudioProcessor) -> None:
        with self._streams_lock:
            self.processor = processor

    def get_debug_snapshot(self) -> dict[str, Any]:
        with self._streams_lock:
            output_stream = self._output_stream
            processor = self.processor
            return {
                "running": self._running,
                "captureBackend": "windows_loopback_soundcard",
                "loopbackEndpointId": self.loopback_endpoint_id,
                "playbackDevice": self.playback_device,
                "samplerate": self.samplerate,
                "channels": self.channels,
                "blocksize": self.blocksize,
                "dtype": self.dtype,
                "latency": self.latency,
                "processorClass": processor.__class__.__name__ if processor is not None else None,
                "outputStreamActive": bool(getattr(output_stream, "active", False)) if output_stream is not None else False,
                "inputQueueSize": self._capture_queue.qsize(),
                "playbackQueueSize": self._playback_queue.qsize(),
                "monitorQueueSize": self._monitor_queue.qsize(),
                "xruns": dict(self.xruns),
            }

    def start(self) -> None:
        with self._streams_lock:
            if self._running:
                return
            self._worker_stop.clear()
            self._capture_stop.clear()
            self._capture_ready.clear()
            self._capture_error = None
            self._diagnostic_chunk_index = 0
            self._consecutive_silent_chunks = 0
            self._capture_anomaly_logs = 0

            try:
                self._worker = threading.Thread(target=self._worker_loop, name="audio-worker", daemon=True)
                self._worker.start()
                self._capture_worker = threading.Thread(target=self._capture_loop, name="audio-loopback-capture", daemon=True)
                self._capture_worker.start()
                if not self._capture_ready.wait(timeout=2.0):
                    self._capture_stop.set()
                    raise RuntimeError("Windows loopback capture thread did not initialize in time")
                if self._capture_error is not None:
                    self._capture_stop.set()
                    raise RuntimeError(str(self._capture_error)) from self._capture_error

                self._output_stream = sd.OutputStream(
                    device=self.playback_device,
                    samplerate=self.samplerate,
                    blocksize=self.blocksize,
                    channels=self.channels,
                    dtype=self.dtype,
                    latency=self.latency,
                    callback=self._output_callback,
                )
                self._output_stream.start()
                self._running = True
            except Exception:
                self._capture_stop.set()
                self._worker_stop.set()
                raise

    def stop(self) -> None:
        with self._streams_lock:
            self._capture_stop.set()
            self._worker_stop.set()

            if self._output_stream is not None:
                try:
                    self._output_stream.stop()
                finally:
                    self._output_stream.close()
            self._output_stream = None
            self._running = False

        if self._capture_worker is not None:
            self._capture_worker.join(timeout=2.0)
            self._capture_worker = None
        if self._worker is not None:
            self._worker.join(timeout=1.0)
            self._worker = None

        self._drain_queue(self._capture_queue)
        self._drain_queue(self._playback_queue)

    def close(self) -> None:
        self.stop()

    def __enter__(self) -> "WindowsLoopbackAudioEngine":
        self.start()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.stop()

    def iter_monitor(self) -> Iterator[np.ndarray]:
        while self._running or not self._monitor_queue.empty():
            try:
                yield self._monitor_queue.get(timeout=0.2)
            except queue.Empty:
                if not self._running:
                    break

    def _capture_loop(self) -> None:
        ole32 = ctypes.windll.ole32
        first_record_logged = False
        ole32.CoInitialize(None)
        try:
            self._logger.info(
                "capture worker starting: endpoint_id=%s samplerate=%s channels=%s blocksize=%s playback_device=%s",
                self.loopback_endpoint_id,
                self.samplerate,
                self.channels,
                self.blocksize,
                self.playback_device,
            )
            soundcard_spec = find_spec("soundcard")
            microphone = sc.get_microphone(self.loopback_endpoint_id, include_loopback=True)
            if microphone is None:
                raise RuntimeError(f"Loopback endpoint is unavailable: {self.loopback_endpoint_id}")
            if not getattr(microphone, "isloopback", False):
                raise RuntimeError(f"Selected endpoint is not a loopback capture source: {self.loopback_endpoint_id}")
            self._logger.info(
                "loopback capture runtime: endpoint_id=%s microphone_id=%s microphone_name=%s is_loopback=%s soundcard_module=%s soundcard_spec_origin=%s",
                self.loopback_endpoint_id,
                getattr(microphone, "id", None),
                getattr(microphone, "name", None),
                bool(getattr(microphone, "isloopback", False)),
                getattr(sc, "__file__", None),
                getattr(soundcard_spec, "origin", None),
            )
            self._logger.info(
                "opening loopback recorder: endpoint_id=%s samplerate=%s channels=%s blocksize=%s",
                self.loopback_endpoint_id,
                self.samplerate,
                self.channels,
                self.blocksize,
            )

            with microphone.recorder(
                samplerate=self.samplerate,
                channels=self.channels,
                blocksize=self.blocksize,
            ) as recorder:
                self._logger.info(
                    "loopback recorder opened: recorder_type=%s endpoint_id=%s",
                    recorder.__class__.__name__,
                    self.loopback_endpoint_id,
                )
                self._capture_ready.set()
                self._logger.info("entering capture loop: endpoint_id=%s", self.loopback_endpoint_id)
                while not self._capture_stop.is_set():
                    if not first_record_logged:
                        self._logger.info(
                            "before first record call: endpoint_id=%s numframes=%s",
                            self.loopback_endpoint_id,
                            self.blocksize,
                        )
                    chunk = recorder.record(numframes=self.blocksize)
                    if not first_record_logged:
                        self._log_first_record_result(chunk)
                        first_record_logged = True
                    self._log_record_anomalies(chunk)
                    chunk = self._normalize_chunk(chunk)
                    self._log_capture_chunk_diagnostics(chunk)
                    self._put_drop_oldest(self._capture_queue, chunk, xrun_key="input")
        except BaseException as exc:
            self._capture_error = exc
            self._capture_ready.set()
            self._logger.exception(
                "capture worker failed: type=%s message=%s traceback=%s",
                type(exc).__name__,
                exc,
                traceback.format_exc(),
            )
            raise
        finally:
            self._logger.info(
                "capture worker stopping: endpoint_id=%s stop_requested=%s chunks_logged=%s consecutive_silent=%s",
                self.loopback_endpoint_id,
                self._capture_stop.is_set(),
                self._diagnostic_chunk_index,
                self._consecutive_silent_chunks,
            )
            ole32.CoUninitialize()
            self._logger.info("capture worker stopped: endpoint_id=%s", self.loopback_endpoint_id)

    def _normalize_chunk(self, chunk: np.ndarray) -> np.ndarray:
        data = np.asarray(chunk, dtype=np.float32)
        if data.ndim == 1:
            data = data[:, None]
        if data.shape[1] == self.channels:
            return data
        if data.shape[1] == 1 and self.channels == 2:
            return np.repeat(data, 2, axis=1)
        if data.shape[1] > self.channels:
            return data[:, :self.channels]
        if data.shape[1] == 0:
            return np.zeros((data.shape[0], self.channels), dtype=np.float32)
        repeats = (self.channels + data.shape[1] - 1) // data.shape[1]
        tiled = np.tile(data, (1, repeats))
        return tiled[:, :self.channels]

    def _log_capture_chunk_diagnostics(self, chunk: np.ndarray) -> None:
        frame_count = int(chunk.shape[0]) if chunk.ndim > 0 else 0
        if frame_count <= 0:
            rms = 0.0
            peak = 0.0
        else:
            rms = float(np.sqrt(np.mean(np.square(chunk, dtype=np.float32), dtype=np.float64)))
            peak = float(np.max(np.abs(chunk)))
        is_silent_chunk = rms <= SILENCE_RMS_THRESHOLD and peak <= SILENCE_PEAK_THRESHOLD
        if is_silent_chunk:
            self._consecutive_silent_chunks += 1
        else:
            self._consecutive_silent_chunks = 0

        chunk_index = self._diagnostic_chunk_index
        self._diagnostic_chunk_index += 1
        should_log = chunk_index < INITIAL_DIAGNOSTIC_CHUNKS
        if not should_log and is_silent_chunk and self._consecutive_silent_chunks >= INITIAL_DIAGNOSTIC_CHUNKS:
            should_log = ((self._consecutive_silent_chunks - INITIAL_DIAGNOSTIC_CHUNKS) % PERSISTENT_SILENCE_LOG_EVERY) == 0
        if not should_log:
            return
        self._logger.info(
            "loopback chunk diag: idx=%s frames=%s rms=%.6f peak=%.6f silent=%s consecutive_silent=%s",
            chunk_index,
            frame_count,
            rms,
            peak,
            is_silent_chunk,
            self._consecutive_silent_chunks,
        )

    def _log_first_record_result(self, chunk: Any) -> None:
        if chunk is None:
            self._logger.warning("after first record call: returned=None")
            return
        shape = getattr(chunk, "shape", None)
        dtype = getattr(chunk, "dtype", None)
        ndim = getattr(chunk, "ndim", None)
        frame_count = int(shape[0]) if shape is not None and len(shape) > 0 else None
        self._logger.info(
            "after first record call: type=%s shape=%s ndim=%s dtype=%s frames=%s",
            type(chunk).__name__,
            shape,
            ndim,
            dtype,
            frame_count,
        )

    def _log_record_anomalies(self, chunk: Any) -> None:
        if self._capture_anomaly_logs >= MAX_CAPTURE_ANOMALY_LOGS:
            return
        if chunk is None:
            self._capture_anomaly_logs += 1
            self._logger.warning("loopback record anomaly: returned None")
            return
        shape = getattr(chunk, "shape", None)
        ndim = getattr(chunk, "ndim", None)
        if shape is None or ndim is None:
            self._capture_anomaly_logs += 1
            self._logger.warning("loopback record anomaly: unexpected non-array result type=%s", type(chunk).__name__)
            return
        if ndim not in (1, 2):
            self._capture_anomaly_logs += 1
            self._logger.warning("loopback record anomaly: unexpected shape=%s ndim=%s", shape, ndim)
            return
        frame_count = int(shape[0]) if len(shape) > 0 else 0
        if frame_count <= 0:
            self._capture_anomaly_logs += 1
            self._logger.warning("loopback record anomaly: zero frames shape=%s ndim=%s", shape, ndim)
            return
        if ndim == 2 and len(shape) > 1 and int(shape[1]) <= 0:
            self._capture_anomaly_logs += 1
            self._logger.warning("loopback record anomaly: empty channel axis shape=%s", shape)

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
