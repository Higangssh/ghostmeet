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
    """After a blind cut the overlap re-feeds audio; it must not re-emit the same speech.

    Speech filling a whole window has no pause to resume from, so the cursor is cut at
    the window edge and the next pass rewinds by the overlap.
    """
    whisper = Whisper(
        script=[
            [RawSeg(1.0, 29.5, "hello there")],
            [RawSeg(0.5, 1.5, "hello there"), RawSeg(3.0, 5.0, "brand new line")],
        ]
    )
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(30.0)
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


# --- window boundaries -------------------------------------------------------------
#
# Regression: on Korean read speech the live path (10 s passes) scored CER ~0.25-0.31
# against ~0.02-0.06 per utterance, on every engine, because speech was duplicated at
# window boundaries. A segment cut off by the window edge was committed as final, and
# the next window re-read the overlap and committed the same speech again.


class TimelineStore(FakeStore):
    """Audio whose sample values are their own absolute index, so a fake model can tell
    exactly which slice of the session it was handed."""

    def read(self, start_sample: int, end_sample: int | None = None) -> np.ndarray:
        end = self.total_samples if end_sample is None else min(end_sample, self.total_samples)
        start = max(0, min(start_sample, end))
        return np.arange(start, end, dtype=np.float64)


class TimelineWhisper:
    """Transcribes a fixed script of utterances the way Whisper does at window edges.

    Speech running past the end of the window comes back cut ("text~cut"); speech that
    began before the window comes back as a fragment ("~text"). A correct transcript
    contains neither marker and each utterance exactly once.
    """

    def __init__(self, utterances):
        self.utterances = utterances  # (absolute start, absolute end, text)
        self.calls: list[int] = []

    def __call__(self, audio: np.ndarray) -> list[RawSeg]:
        self.calls.append(len(audio))
        ws = audio[0] / SAMPLE_RATE
        we = (audio[-1] + 1) / SAMPLE_RATE
        out = []
        for start, end, text in self.utterances:
            if end <= ws or start >= we:
                continue
            label = ("~" if start < ws else "") + text + ("~cut" if end > we else "")
            out.append(RawSeg(max(start, ws) - ws, min(end, we) - ws, label))
        return out


def run_live(store, transcriber, seconds, tick=10.0):
    for _ in range(int(seconds / tick)):
        store.grow(tick)
        transcriber.step()
    for _ in range(10):
        if transcriber.pending_samples == 0:
            break
        transcriber.step(force=True)


def test_speech_cut_by_the_window_edge_is_not_committed_twice():
    """Live passes: a sentence still being spoken at the edge waits for the next pass."""
    script = [(1.0, 6.0, "one"), (7.0, 13.0, "two"), (14.0, 18.5, "three"), (21.0, 28.0, "four")]
    store = TimelineStore()
    transcriber = IncrementalTranscriber(store, TimelineWhisper(script), window_sec=30.0, overlap_sec=2.0)

    run_live(store, transcriber, 30.0)

    assert [s.text for s in transcriber.transcript] == ["one", "two", "three", "four"]
    assert [(s.start, s.end) for s in transcriber.transcript] == [(a, b) for a, b, _ in script]


def test_a_long_segment_spanning_the_overlap_is_not_re_emitted():
    """Backlog passes: the sentence crossing the 30 s edge appears once, whole."""
    script = [(2.0, 12.0, "a"), (14.0, 26.0, "b"), (27.0, 35.0, "c"), (37.0, 50.0, "d")]
    store = TimelineStore()
    transcriber = IncrementalTranscriber(store, TimelineWhisper(script), window_sec=30.0, overlap_sec=2.0)

    store.grow(60.0)
    for _ in range(10):
        if transcriber.pending_samples == 0:
            break
        transcriber.step(force=True)

    assert [s.text for s in transcriber.transcript] == ["a", "b", "c", "d"]
    assert transcriber.pending_samples == 0


def test_a_held_back_tail_is_committed_by_the_final_forced_pass():
    """Holding back the edge must never lose it: the session's last words still land."""
    script = [(1.0, 5.0, "x"), (6.0, 9.8, "y")]
    store = TimelineStore()
    transcriber = IncrementalTranscriber(store, TimelineWhisper(script), window_sec=30.0, overlap_sec=2.0)

    store.grow(10.0)
    assert [s.text for s in transcriber.step()] == ["x"]  # y runs into the edge
    assert [s.text for s in transcriber.step(force=True)] == ["y"]
    assert transcriber.pending_samples == 0


def test_speech_longer_than_a_window_still_makes_progress_within_bounds():
    """Holding back must not stall a cursor when no pause ever comes."""
    store = TimelineStore()
    whisper = TimelineWhisper([(0.0, 100.0, "monologue")])
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    store.grow(100.0)
    for _ in range(20):
        if transcriber.pending_samples == 0:
            break
        transcriber.step(force=True)

    assert transcriber.pending_samples == 0
    assert transcriber.transcript, "the monologue was dropped instead of committed"
    assert max(whisper.calls) <= 30 * SAMPLE_RATE


def test_holding_back_keeps_the_per_pass_window_bounded_for_four_hours():
    """Speech that always runs into the edge must not grow the window past its cap."""
    store = TimelineStore()
    script = [(t + 1.0, t + 10.0, f"s{i}") for i, t in enumerate(range(0, 4 * 3600, 10))]
    whisper = TimelineWhisper(script)
    transcriber = IncrementalTranscriber(store, whisper, window_sec=30.0, overlap_sec=2.0)

    run_live(store, transcriber, 4 * 3600.0)

    assert max(whisper.calls) <= 30 * SAMPLE_RATE
    assert len(transcriber.transcript) == len(script)
    assert all("~" not in s.text for s in transcriber.transcript)
