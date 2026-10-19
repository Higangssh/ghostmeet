"""Incremental webm/opus decoding for a live capture stream.

MediaRecorder emits webm where only the first chunk carries the EBML header, so
individual chunks cannot be decoded on their own. Rather than re-reading the whole
growing file on every transcription pass (cost grows with session length), we keep a
single demuxer open over the stream for the life of the session and push PCM out as
frames arrive. Cost stays linear, which is what long meetings require.
"""
from __future__ import annotations

import io
import logging
import queue
import threading
from typing import Protocol

import av

logger = logging.getLogger(__name__)

TARGET_SAMPLE_RATE = 16000
_PROBE_BUFFER_BYTES = 32 * 1024


class PcmSink(Protocol):
    def append(self, pcm_bytes: bytes) -> None: ...


class _QueueReader(io.RawIOBase):
    """File-like view over a byte queue, so PyAV can pull from a live socket."""

    def __init__(self, source: queue.Queue):
        self._source = source
        self._buf = memoryview(b"")
        self._eof = False

    def readable(self) -> bool:
        return True

    def readinto(self, target) -> int:  # noqa: ANN001 - buffer protocol
        while not self._buf:
            if self._eof:
                return 0
            chunk = self._source.get()
            if chunk is None:
                self._eof = True
                return 0
            self._buf = memoryview(chunk)

        count = min(len(target), len(self._buf))
        target[:count] = self._buf[:count]
        self._buf = self._buf[count:]
        return count


class StreamingWebmDecoder:
    """Decodes a growing webm/opus stream into 16 kHz mono s16le PCM as it arrives."""

    def __init__(self, sink: PcmSink, sample_rate: int = TARGET_SAMPLE_RATE):
        self._sink = sink
        self._sample_rate = sample_rate
        self._queue: queue.Queue = queue.Queue()
        self._closing = threading.Event()
        self.error: BaseException | None = None
        self._thread = threading.Thread(
            target=self._run, name="ghostmeet-decoder", daemon=True
        )
        self._thread.start()

    def feed(self, chunk: bytes) -> None:
        """Hand raw container bytes to the decoder. Never blocks the caller."""
        if self._closing.is_set() or not chunk:
            return
        self._queue.put(chunk)

    def close(self, timeout: float = 15.0) -> None:
        """Signal end of stream and wait for the decoder to drain."""
        if not self._closing.is_set():
            self._closing.set()
            self._queue.put(None)
        self._thread.join(timeout)
        if self._thread.is_alive():
            logger.warning("Decoder thread did not exit within %.1fs", timeout)

    @property
    def running(self) -> bool:
        return self._thread.is_alive()

    def _run(self) -> None:
        try:
            reader = _QueueReader(self._queue)
            with av.open(reader, format="webm", buffer_size=_PROBE_BUFFER_BYTES) as container:
                resampler = av.AudioResampler(
                    format="s16", layout="mono", rate=self._sample_rate
                )
                for frame in container.decode(audio=0):
                    self._emit(resampler.resample(frame))
                self._emit(resampler.resample(None))  # flush
        except BaseException as exc:  # noqa: BLE001 - recorded, not swallowed
            self.error = exc
            logger.error("Audio decoding stopped: %s", exc, exc_info=True)

    def _emit(self, frames) -> None:
        for frame in frames:
            if frame is None:
                continue
            self._sink.append(frame.to_ndarray().astype("<i2").tobytes())
