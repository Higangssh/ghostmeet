"""IncrementalTranscriber — windowed transcription with absolute timestamps.

This is the component that replaces "re-transcribe the whole accumulated file every
interval". The invariant that matters: the amount of audio handed to Whisper per pass
is bounded by the window, no matter how long the meeting has been running.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pytest

from backend.incremental import IncrementalTranscriber, Segment

SAMPLE_RATE = 16000


@dataclass
class RawSeg:
    """Mimics a faster-whisper segment: times are relative to the audio passed in."""

    start: float
    end: float
    text: str


class FakeStore:
    """Duck-types PcmStore without putting hours of audio on disk."""

    def __init__(self):
        self.total_samples = 0

    def grow(self, seconds: float) -> None:
        self.total_samples += int(seconds * SAMPLE_RATE)

    def read(self, start_sample: int, end_sample: int | None = None) -> np.ndarray:
        end = self.total_samples if end_sample is None else min(end_sample, self.total_samples)
        start = max(0, min(start_sample, end))
        return np.zeros(end - start, dtype=np.float32)


class Whisper:
    """Records every call so tests can assert on how much audio was processed."""

    def __init__(self, script=None):
        self.calls: list[int] = []
        self._script = list(script or [])

    def __call__(self, audio: np.ndarray) -> list[RawSeg]:
        self.calls.append(len(audio))
        return self._script.pop(0) if self._script else []

    @property
    def largest_call_samples(self) -> int:
        return max(self.calls, default=0)


@pytest.fixture
def store():
    return FakeStore()


def test_no_audio_means_no_whisper_call(store):
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper)

    assert transcriber.step() == []
    assert whisper.calls == []


def test_skips_whisper_until_enough_new_audio_accumulates(store):
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, min_new_sec=1.0)

    store.grow(0.4)
    assert transcriber.step() == []
    assert whisper.calls == []

    store.grow(0.7)  # now 1.1s total
    transcriber.step()
    assert len(whisper.calls) == 1


def test_segment_times_are_absolute_not_window_relative(store):
    """A segment found 1s into a window that starts at 60s must report 61s."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(60.0)
    for _ in range(20):  # drain the first silent minute
        if transcriber.pending_samples == 0:
            break
        transcriber.step()

    whisper._script = [[RawSeg(1.0, 3.0, "second window")]]
    store.grow(10.0)
    found = transcriber.step()

    assert len(found) == 1
    window_start = 60.0 - 2.0  # cursor rewound by the overlap
    assert found[0].start == pytest.approx(window_start + 1.0)
    assert found[0].end == pytest.approx(window_start + 3.0)


def test_audio_per_pass_stays_bounded_across_a_four_hour_session(store):
    """The regression test for the O(n^2) rewrite: cost per pass must not grow."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    for _ in range(1440):  # 4 hours in 10s ticks
        store.grow(10.0)
        transcriber.step()

    assert store.total_samples == 4 * 3600 * SAMPLE_RATE
    assert len(whisper.calls) == 1440  # linear in session length, one pass per tick
    assert whisper.largest_call_samples <= 30 * SAMPLE_RATE


def test_a_stalled_cursor_cannot_grow_the_window_without_bound(store):
    """Even if Whisper returns nothing for hours, the window must stay clamped."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0)

    store.grow(3600.0)  # an hour arrives before the first pass
    transcriber.step()

    assert whisper.largest_call_samples <= 30 * SAMPLE_RATE


def test_a_backlog_is_caught_up_oldest_first_and_never_skipped(store):
    """Regression: anchoring the window to the newest audio marked the middle of a
    backlog as consumed without transcribing it, silently losing meeting content."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(90.0)  # three windows arrive before the first pass runs
    transcriber.step()

    assert transcriber.pending_samples > 0, "jumped to the newest audio, skipping the rest"

    for _ in range(20):
        if transcriber.pending_samples == 0:
            break
        transcriber.step()

    assert transcriber.pending_samples == 0
    assert sum(whisper.calls) >= 90 * SAMPLE_RATE  # every second reached Whisper


def test_forced_pass_transcribes_a_trailing_fraction_of_a_second(store):
    """A meeting ending mid-word must not drop its final moments."""
    whisper = Whisper(script=[[RawSeg(0.0, 0.4, "bye")]])
    transcriber = IncrementalTranscriber(store, whisper, min_new_sec=1.0)

    store.grow(0.4)

    assert transcriber.step() == []  # below the normal threshold
    assert [s.text for s in transcriber.step(force=True)] == ["bye"]


def test_forced_pass_on_an_empty_session_does_nothing(store):
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper)

    assert transcriber.step(force=True) == []
    assert whisper.calls == []


def test_overlapping_audio_does_not_produce_duplicate_segments(store):
    """The overlap re-feeds audio to Whisper; it must not re-emit the same speech."""
    whisper = Whisper(
        script=[
            [RawSeg(1.0, 4.0, "hello there")],
            [RawSeg(0.5, 1.5, "hello there"), RawSeg(3.0, 5.0, "brand new line")],
        ]
    )
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(10.0)
    first = transcriber.step()
    store.grow(10.0)
    second = transcriber.step()

    assert [s.text for s in first] == ["hello there"]
    assert [s.text for s in second] == ["brand new line"]
    assert [s.text for s in transcriber.transcript] == ["hello there", "brand new line"]


def test_speech_cut_off_at_the_window_edge_is_retried(store):
    """Trailing audio after the last segment is re-read, so cut words get a second pass."""
    whisper = Whisper(script=[[RawSeg(0.0, 5.0, "complete sentence")], []])
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(10.0)
    transcriber.step()
    store.grow(1.5)
    transcriber.step()

    # second pass re-reads from the end of the last segment (5s), not from 10s
    assert whisper.calls[1] > int(1.5 * SAMPLE_RATE)


def test_silence_advances_the_cursor_so_it_is_not_rescanned_forever(store):
    """A quiet meeting must not re-scan the same audio on every pass.

    Regression: when the cursor failed to advance through silence, every later pass hit
    the window clamp and burned a full 30s of Whisper work for 5s of new audio.
    """
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    for _ in range(20):
        store.grow(5.0)
        transcriber.step()

    # every pass should see the new 5s plus the 2s overlap — never the whole window
    assert whisper.calls[-1] == pytest.approx(7.0 * SAMPLE_RATE, rel=0.01)
    assert whisper.largest_call_samples < 30 * SAMPLE_RATE


def test_segments_expose_a_speaker_slot_for_diarization(store):
    """Speaker attribution is the top request on the thread; reserve the field now."""
    whisper = Whisper(script=[[RawSeg(0.0, 1.0, "who said this")]])
    transcriber = IncrementalTranscriber(store, whisper)

    store.grow(5.0)
    segment = transcriber.step()[0]

    assert segment.speaker is None
    assert "speaker" in segment.to_dict()


def test_transcript_and_full_text_preserve_order(store):
    whisper = Whisper(
        script=[[RawSeg(0.0, 1.0, "first")], [RawSeg(0.0, 1.0, "second")]]
    )
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=0.0)

    store.grow(10.0)
    transcriber.step()
    store.grow(10.0)
    transcriber.step()

    assert transcriber.full_text() == "first second"
    assert [s.text for s in transcriber.transcript] == ["first", "second"]


def test_blank_segments_from_whisper_are_dropped(store):
    whisper = Whisper(script=[[RawSeg(0.0, 1.0, "   "), RawSeg(1.0, 2.0, "real text")]])
    transcriber = IncrementalTranscriber(store, whisper)

    store.grow(5.0)
    found = transcriber.step()

    assert [s.text for s in found] == ["real text"]


def test_timestamps_remain_accurate_after_four_hours(store):
    """Long sessions must not drift: absolute time comes from the sample cursor."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    for _ in range(1439):
        store.grow(10.0)
        transcriber.step()

    whisper._script = [[RawSeg(9.0, 9.5, "final word")]]
    store.grow(10.0)
    found = transcriber.step()

    # Drift would show up as an absolute time that no longer matches the audio ingested:
    # the final segment has to land inside the session, within the last window.
    duration = store.total_samples / SAMPLE_RATE
    assert duration == 4 * 3600
    assert found[0].end <= duration
    assert found[0].end > duration - 30.0


def test_whisper_failure_does_not_lose_the_session(store):
    """A bad pass must not kill an in-progress meeting or stall the cursor."""

    def exploding(audio):
        raise RuntimeError("model blew up")

    transcriber = IncrementalTranscriber(store, exploding)
    store.grow(10.0)

    assert transcriber.step() == []
    assert transcriber.transcript == []


def test_pending_samples_reports_audio_not_yet_consumed(store):
    """Used at session end to drain trailing audio instead of dropping it."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0)

    assert transcriber.pending_samples == 0

    store.grow(45.0)
    assert transcriber.pending_samples == int(45.0 * SAMPLE_RATE)

    transcriber.step()  # one pass can only consume one window

    assert transcriber.pending_samples > 0
    assert transcriber.pending_samples < int(45.0 * SAMPLE_RATE)


def test_repeated_steps_drain_a_long_backlog_to_zero(store):
    """A 4-hour recording still fully processes if passes run back to back."""
    whisper = Whisper()
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0)
    store.grow(600.0)

    for _ in range(100):
        if transcriber.pending_samples == 0:
            break
        transcriber.step()

    assert transcriber.pending_samples == 0
    assert whisper.largest_call_samples <= 30 * SAMPLE_RATE


def test_segment_dict_round_trip_has_the_fields_the_ui_uses(store):
    segment = Segment(text="hi", start=1.0, end=2.0)

    data = segment.to_dict()

    assert data["text"] == "hi"
    assert data["start"] == 1.0
    assert data["end"] == 2.0
    assert "timestamp" in data
