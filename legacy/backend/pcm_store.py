"""Append-only PCM buffer backed by a file.

Audio accumulates on disk so a session's memory footprint stays flat regardless of
length — transcription only ever reads a short tail window back into RAM.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

SAMPLE_RATE = 16000
BYTES_PER_SAMPLE = 2  # s16le
INT16_FULL_SCALE = 32768.0


class PcmStore:
    """16 kHz mono s16le PCM on disk, appended incrementally and read by window."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write = self.path.open("ab")
        self._bytes_written = self.path.stat().st_size

    @property
    def total_samples(self) -> int:
        """Whole samples available. A dangling odd byte is not yet a sample."""
        return self._bytes_written // BYTES_PER_SAMPLE

    @property
    def duration(self) -> float:
        return self.total_samples / SAMPLE_RATE

    def append(self, pcm_bytes: bytes) -> None:
        if not pcm_bytes or self._write.closed:
            # a decoder thread can still be draining when the session tears down;
            # dropping those bytes is correct, crashing on them is not
            return
        self._write.write(pcm_bytes)
        self._write.flush()
        self._bytes_written += len(pcm_bytes)

    def read(self, start_sample: int, end_sample: int | None = None) -> np.ndarray:
        """Return samples [start, end) as float32 in [-1, 1), clamped to what exists."""
        total = self.total_samples
        start = max(0, min(start_sample, total))
        end = total if end_sample is None else max(start, min(end_sample, total))
        count = end - start
        if count <= 0:
            return np.empty(0, dtype=np.float32)

        with self.path.open("rb") as fh:
            fh.seek(start * BYTES_PER_SAMPLE)
            raw = fh.read(count * BYTES_PER_SAMPLE)

        return np.frombuffer(raw, dtype="<i2").astype(np.float32) / INT16_FULL_SCALE

    def close(self) -> None:
        if not self._write.closed:
            self._write.close()
