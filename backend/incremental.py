"""Windowed incremental transcription.

Each pass transcribes only the audio that has arrived since the last one, plus a small
overlap so speech straddling a window boundary still gets a clean read. The window has
a hard upper bound, which is what keeps a 4-hour meeting costing the same per pass as
the first minute.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Callable, List, Protocol, Sequence

import numpy as np

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class Segment:
    """One transcribed span, timed from the start of the session."""

    text: str
    start: float
    end: float
    speaker: str | None = None  # filled in once diarization lands
    timestamp: float = field(default_factory=time.time)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "start": round(self.start, 2),
            "end": round(self.end, 2),
            "speaker": self.speaker,
            "timestamp": self.timestamp,
        }


class AudioSource(Protocol):
    @property
    def total_samples(self) -> int: ...
    def read(self, start_sample: int, end_sample: int | None = None) -> np.ndarray: ...


class RawSegment(Protocol):
    start: float
    end: float
    text: str


TranscribeFn = Callable[[np.ndarray], Sequence[RawSegment]]


class IncrementalTranscriber:
    """Transcribes a growing audio source one bounded window at a time."""

    def __init__(
        self,
        source: AudioSource,
        transcribe_fn: TranscribeFn,
        window_sec: float = 30.0,
        overlap_sec: float = 2.0,
        min_new_sec: float = 1.0,
    ):
        self._source = source
        self._transcribe = transcribe_fn
        self._window_samples = int(window_sec * SAMPLE_RATE)
        self._overlap_samples = int(overlap_sec * SAMPLE_RATE)
        self._min_new_samples = int(min_new_sec * SAMPLE_RATE)
        self._cursor = 0  # samples consumed
        self._emitted_until = 0.0  # absolute seconds covered by emitted segments
        self.transcript: List[Segment] = []

    @property
    def pending_samples(self) -> int:
        """Audio that has arrived but not been consumed yet."""
        return max(0, self._source.total_samples - self._cursor)

    def step(self, force: bool = False) -> List[Segment]:
        """Transcribe the oldest audio not yet consumed.

        Set `force` for the final pass of a session, so a meeting that ends mid-word
        still gets its last fraction of a second transcribed.
        """
        total = self._source.total_samples
        pending = total - self._cursor
        if pending <= 0 or (not force and pending < self._min_new_samples):
            return []

        # Oldest-first, so a backlog is caught up rather than skipped over. Bounding the
        # END (not the start) keeps cost per pass flat without ever dropping audio.
        start = max(0, self._cursor - self._overlap_samples)
        end = min(total, start + self._window_samples)

        audio = self._source.read(start, end)
        if audio.size == 0:
            return []

        offset = start / SAMPLE_RATE
        try:
            raw = list(self._transcribe(audio))
        except Exception as exc:  # noqa: BLE001 - one bad pass must not end the meeting
            logger.error("Transcription pass failed at %.1fs: %s", offset, exc, exc_info=True)
            self._cursor = end
            return []

        new = self._accept(raw, offset)
        self._advance_cursor(end, previous_cursor=self._cursor)
        return new

    def _accept(self, raw: Sequence[RawSegment], offset: float) -> List[Segment]:
        """Keep segments whose midpoint falls after everything already emitted."""
        new: List[Segment] = []
        for item in raw:
            text = item.text.strip()
            if not text:
                continue
            start = offset + item.start
            end = offset + item.end
            if (start + end) / 2 <= self._emitted_until:
                continue
            segment = Segment(text=text, start=start, end=end)
            self.transcript.append(segment)
            new.append(segment)
            self._emitted_until = max(self._emitted_until, end)
        return new

    def _advance_cursor(self, window_end: int, previous_cursor: int) -> None:
        """Resume from the end of known speech, so cut-off words get another pass."""
        resume = int(self._emitted_until * SAMPLE_RATE)
        self._cursor = min(window_end, max(previous_cursor, resume))
        if self._cursor <= previous_cursor:
            # nothing new was recognised — consume the window rather than rescan it
            self._cursor = window_end

    def full_text(self) -> str:
        return " ".join(s.text for s in self.transcript)

    def snapshot(self) -> List[dict]:
        return [s.to_dict() for s in self.transcript]
