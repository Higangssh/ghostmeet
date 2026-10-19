"""Windowed incremental transcription.

Each pass transcribes only the audio that has arrived since the last one. The window has
a hard upper bound, which is what keeps a 4-hour meeting costing the same per pass as
the first minute.

Window edges are where duplicates come from. A segment that ends near the edge may be a
sentence cut off mid-word, so it is held back rather than committed, and the next pass
starts exactly where the last committed segment ended. That boundary is a real pause,
so nothing is re-read and nothing can be emitted twice. Only when the cursor had to be
cut blindly (silence, or speech filling a whole window) does the next pass rewind by
`overlap_sec` to catch words straddling the cut.
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
        holdback_sec: float = 1.0,
    ):
        self._source = source
        self._transcribe = transcribe_fn
        self._window_samples = int(window_sec * SAMPLE_RATE)
        self._overlap_samples = int(overlap_sec * SAMPLE_RATE)
        self._min_new_samples = int(min_new_sec * SAMPLE_RATE)
        self._holdback = holdback_sec
        self._cursor = 0  # samples consumed
        self._blind_cut = False  # cursor sits at a window edge, not at the end of speech
        self._emitted_until = 0.0  # absolute seconds covered by emitted segments
        self.transcript: List[Segment] = []

    @property
    def pending_samples(self) -> int:
        """Audio that has arrived but not been consumed yet."""
        return max(0, self._source.total_samples - self._cursor)

    def step(self, force: bool = False) -> List[Segment]:
        """Transcribe the oldest audio not yet consumed.

        Set `force` once no more audio is coming: the threshold is skipped and speech
        running into the end of the session is committed instead of held back.
        """
        total = self._source.total_samples
        pending = total - self._cursor
        if pending <= 0 or (not force and pending < self._min_new_samples):
            return []

        # Oldest-first, so a backlog is caught up rather than skipped over. Bounding the
        # END (not the start) keeps cost per pass flat without ever dropping audio.
        rewind = self._overlap_samples if self._blind_cut else 0
        start = max(0, self._cursor - rewind)
        end = min(total, start + self._window_samples)

        audio = self._source.read(start, end)
        if audio.size == 0:
            return []

        offset = start / SAMPLE_RATE
        try:
            raw = list(self._transcribe(audio))
        except Exception as exc:  # noqa: BLE001 - one bad pass must not end the meeting
            logger.error("Transcription pass failed at %.1fs: %s", offset, exc, exc_info=True)
            self._cursor, self._blind_cut = end, True
            return []

        spoken = [(offset + r.start, offset + r.end, r.text.strip()) for r in raw if r.text.strip()]
        final = force and end == total  # no audio after this window will ever be read
        ready = self._committable(spoken, end / SAMPLE_RATE, final)
        blind = final
        if not ready and spoken and end - start >= self._window_samples:
            # speech fills the whole window: there is no pause to wait for, so cut blindly
            ready, blind = spoken, True
        held = len(ready) < len(spoken)

        previous = self._cursor
        new = self._accept(ready)
        if blind:
            self._cursor, self._blind_cut = end, True
        elif new:
            # resume exactly at the end of committed speech: a real pause, nothing re-read
            self._cursor = min(end, max(previous, int(self._emitted_until * SAMPLE_RATE)))
            self._blind_cut = False
        elif not held:
            # nothing recognised — consume the window rather than rescan it
            self._cursor, self._blind_cut = end, True
        # else: only speech still running into the edge; wait for more audio. The next
        # window starts at the same cursor and is still capped at window_sec.
        return new

    def _committable(self, spoken: List[tuple], window_end: float, final: bool) -> List[tuple]:
        """Segments safe to commit: all of them on the final pass, otherwise those ending
        before the hold-back margin. Order is kept, so the first one held holds the rest."""
        if final:
            return spoken
        edge = window_end - self._holdback
        ready = []
        for seg in spoken:
            if seg[1] > edge:
                break
            ready.append(seg)
        return ready

    def _accept(self, spoken: Sequence[tuple]) -> List[Segment]:
        """Keep segments whose midpoint falls after everything already emitted."""
        new: List[Segment] = []
        for start, end, text in spoken:
            if (start + end) / 2 <= self._emitted_until:
                continue
            segment = Segment(text=text, start=start, end=end)
            self.transcript.append(segment)
            new.append(segment)
            self._emitted_until = max(self._emitted_until, end)
        return new

    def full_text(self) -> str:
        return " ".join(s.text for s in self.transcript)

    def snapshot(self) -> List[dict]:
        return [s.to_dict() for s in self.transcript]
