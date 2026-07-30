"""SessionPipeline — wiring receive, decode, and transcription without blocking.

The old design ran transcription inline in the WebSocket receive loop, so audio chunks
piled up unread for as long as a pass took. Here the socket only ever hands bytes to the
decoder, and transcription runs off the event loop on its own schedule.
"""
from __future__ import annotations

import asyncio
import threading
import time
from dataclasses import dataclass

import numpy as np
import pytest

from backend.incremental import IncrementalTranscriber
from backend.pipeline import SessionPipeline

SAMPLE_RATE = 16000


@dataclass
class RawSeg:
    start: float
    end: float
    text: str


class FakeStore:
    def __init__(self):
        self.total_samples = 0

    def grow(self, seconds: float) -> None:
        self.total_samples += int(seconds * SAMPLE_RATE)

    def read(self, start_sample, end_sample=None):
        end = self.total_samples if end_sample is None else min(end_sample, self.total_samples)
        start = max(0, min(start_sample, end))
        return np.zeros(end - start, dtype=np.float32)


class FakeDecoder:
    """Stands in for StreamingWebmDecoder: each fed chunk becomes a second of audio."""

    def __init__(self, store: FakeStore, seconds_per_chunk: float = 1.0):
        self.store = store
        self.seconds_per_chunk = seconds_per_chunk
        self.fed = 0
        self.closed = False
        self.error = None

    def feed(self, chunk: bytes) -> None:
        if self.closed:
            return
        self.fed += 1
        self.store.grow(self.seconds_per_chunk)

    def close(self, timeout: float = 15.0) -> None:
        self.closed = True


class Whisper:
    def __init__(self, delay: float = 0.0, script=None):
        self.delay = delay
        self.calls = 0
        self.concurrent = 0
        self.max_concurrent = 0
        self._lock = threading.Lock()
        self._script = list(script or [])

    def __call__(self, audio: np.ndarray):
        with self._lock:
            self.calls += 1
            self.concurrent += 1
            self.max_concurrent = max(self.max_concurrent, self.concurrent)
        try:
            if self.delay:
                time.sleep(self.delay)
            return self._script.pop(0) if self._script else []
        finally:
            with self._lock:
                self.concurrent -= 1


class Collector:
    def __init__(self):
        self.batches = []

    async def __call__(self, segments):
        self.batches.append(segments)

    @property
    def texts(self):
        return [s.text for batch in self.batches for s in batch]


def build(whisper, interval=0.02, seconds_per_chunk=1.0, **kwargs):
    store = FakeStore()
    decoder = FakeDecoder(store, seconds_per_chunk)
    incremental = IncrementalTranscriber(store, whisper, **kwargs)
    collector = Collector()
    pipeline = SessionPipeline(
        decoder=decoder,
        transcriber=incremental,
        on_segments=collector,
        interval_sec=interval,
    )
    return pipeline, decoder, collector


async def test_feed_returns_immediately_while_a_pass_is_running():
    """The receive loop must never wait on Whisper."""
    whisper = Whisper(delay=0.5)
    pipeline, _, _ = build(whisper)
    await pipeline.start()
    try:
        pipeline.feed(b"chunk")
        await asyncio.sleep(0.1)  # let a slow pass get underway
        assert whisper.calls >= 1

        started = time.monotonic()
        for _ in range(200):
            pipeline.feed(b"chunk")
        elapsed = time.monotonic() - started

        assert elapsed < 0.2, f"feed blocked for {elapsed:.2f}s"
    finally:
        await pipeline.stop()


async def test_event_loop_keeps_running_during_transcription():
    """Transcription must happen off the loop, or the socket stops being serviced."""
    whisper = Whisper(delay=0.4)
    pipeline, _, _ = build(whisper)
    ticks = 0

    async def ticker():
        nonlocal ticks
        while True:
            await asyncio.sleep(0.01)
            ticks += 1

    await pipeline.start()
    task = asyncio.create_task(ticker())
    try:
        pipeline.feed(b"chunk")
        await asyncio.sleep(0.45)

        assert whisper.calls >= 1
        assert ticks > 10, f"event loop stalled (only {ticks} ticks)"
    finally:
        task.cancel()
        await pipeline.stop()


async def test_transcription_passes_never_overlap():
    """A pass slower than the interval must not stack up on itself."""
    whisper = Whisper(delay=0.15)
    pipeline, _, _ = build(whisper, interval=0.01)
    await pipeline.start()
    try:
        for _ in range(5):
            pipeline.feed(b"chunk")
        await asyncio.sleep(0.6)
    finally:
        await pipeline.stop()

    assert whisper.max_concurrent == 1


async def test_segments_are_broadcast_while_the_session_is_live():
    whisper = Whisper(script=[[RawSeg(0.0, 1.0, "live caption")]])
    pipeline, _, collector = build(whisper, interval=0.02)
    await pipeline.start()
    try:
        pipeline.feed(b"chunk")
        for _ in range(50):
            if collector.texts:
                break
            await asyncio.sleep(0.02)

        assert collector.texts == ["live caption"]
    finally:
        await pipeline.stop()


async def test_stop_transcribes_audio_that_arrived_after_the_last_pass():
    """Trailing audio at hangup must not be dropped."""
    whisper = Whisper(script=[[RawSeg(0.0, 1.0, "last words")]])
    pipeline, _, collector = build(whisper, interval=100.0)  # no periodic pass will fire
    await pipeline.start()

    pipeline.feed(b"chunk")
    await pipeline.stop()

    assert collector.texts == ["last words"]


async def test_stop_drains_a_backlog_larger_than_one_window():
    """If transcription fell behind, stopping must still process everything captured."""
    whisper = Whisper()
    pipeline, _, _ = build(
        whisper, interval=100.0, seconds_per_chunk=30.0, window_sec=30.0
    )
    await pipeline.start()

    for _ in range(5):  # 150s of audio, no pass yet
        pipeline.feed(b"chunk")
    await pipeline.stop()

    assert pipeline.transcriber.pending_samples == 0


async def test_stop_closes_the_decoder():
    whisper = Whisper()
    pipeline, decoder, _ = build(whisper)
    await pipeline.start()

    await pipeline.stop()

    assert decoder.closed is True


async def test_stop_is_safe_when_nothing_was_ever_captured():
    whisper = Whisper()
    pipeline, _, collector = build(whisper)
    await pipeline.start()

    await pipeline.stop()

    assert collector.texts == []
    assert whisper.calls == 0


async def test_stop_is_idempotent():
    whisper = Whisper()
    pipeline, _, _ = build(whisper)
    await pipeline.start()

    await pipeline.stop()
    await pipeline.stop()


async def test_feed_after_stop_is_ignored():
    whisper = Whisper()
    pipeline, decoder, _ = build(whisper)
    await pipeline.start()
    await pipeline.stop()

    pipeline.feed(b"late chunk")

    assert decoder.fed == 0


async def test_a_failing_pass_does_not_kill_the_session():
    """One bad window must not stop later captions from arriving."""

    class Flaky:
        def __init__(self):
            self.calls = 0

        def __call__(self, audio):
            self.calls += 1
            if self.calls == 1:
                raise RuntimeError("bad window")
            return [RawSeg(0.0, 1.0, "recovered")]

    pipeline, _, collector = build(Flaky(), interval=0.02)
    await pipeline.start()
    try:
        pipeline.feed(b"chunk")
        for _ in range(50):
            if collector.texts:
                break
            pipeline.feed(b"chunk")
            await asyncio.sleep(0.02)

        assert "recovered" in collector.texts
    finally:
        await pipeline.stop()


async def test_broadcast_failure_does_not_stop_transcription():
    """A closed side panel must not take the session down with it."""

    async def exploding(segments):
        raise RuntimeError("subscriber went away")

    store = FakeStore()
    decoder = FakeDecoder(store)
    whisper = Whisper(script=[[RawSeg(0.0, 1.0, "one")], [RawSeg(1.0, 2.0, "two")]])
    incremental = IncrementalTranscriber(store, whisper)
    pipeline = SessionPipeline(
        decoder=decoder, transcriber=incremental, on_segments=exploding, interval_sec=0.02
    )
    await pipeline.start()
    try:
        for _ in range(30):
            pipeline.feed(b"chunk")
            await asyncio.sleep(0.02)
            if whisper.calls >= 2:
                break

        assert whisper.calls >= 2
    finally:
        await pipeline.stop()

    assert [s.text for s in incremental.transcript] == ["one", "two"]
