"""StreamingWebmDecoder — incremental webm/opus → 16 kHz mono PCM.

The extension sends webm/opus in ~1s chunks where only the first chunk carries the
EBML header, so chunks are not independently decodable. The decoder therefore keeps
ONE demuxer open over the whole growing stream and emits PCM as it goes. That is what
keeps cost linear: without it, every transcription pass would re-decode the session
from byte zero, which is unusable for the 4+ hour meetings users asked for.
"""
from __future__ import annotations

import io
import time

import av
import numpy as np
import pytest

from backend.decoder import StreamingWebmDecoder

SAMPLE_RATE = 16000


def make_webm_opus(seconds: float = 2.0, freq: float = 440.0) -> bytes:
    """Encode a sine wave as webm/opus, the way MediaRecorder would."""
    rate = 48000
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format="webm")
    stream = container.add_stream("libopus", rate=rate)
    stream.layout = "mono"

    frame_size = 960  # 20 ms at 48 kHz, opus' native frame
    total = int(rate * seconds)
    t = np.arange(total, dtype=np.float32) / rate
    wave = (np.sin(2 * np.pi * freq * t) * 0.8 * 32767).astype("<i2")

    for offset in range(0, total - frame_size + 1, frame_size):
        block = wave[offset : offset + frame_size].reshape(1, -1)
        frame = av.AudioFrame.from_ndarray(block, format="s16", layout="mono")
        frame.rate = rate
        frame.pts = offset
        for packet in stream.encode(frame):
            container.mux(packet)

    for packet in stream.encode(None):
        container.mux(packet)
    container.close()
    return buf.getvalue()


class ListSink:
    """Collects appended PCM bytes, mimicking PcmStore.append."""

    def __init__(self):
        self.chunks: list[bytes] = []

    def append(self, pcm: bytes) -> None:
        self.chunks.append(pcm)

    @property
    def data(self) -> bytes:
        return b"".join(self.chunks)

    @property
    def samples(self) -> int:
        return len(self.data) // 2

    def as_float(self) -> np.ndarray:
        return np.frombuffer(self.data, dtype="<i2").astype(np.float32) / 32768.0


def wait_until(predicate, timeout: float = 15.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if predicate():
            return True
        time.sleep(0.02)
    return False


@pytest.fixture(scope="module")
def webm_two_seconds() -> bytes:
    return make_webm_opus(seconds=2.0)


def test_fixture_is_a_real_webm_stream(webm_two_seconds):
    """Guard the test's own premise before relying on it."""
    assert webm_two_seconds[:4] == b"\x1aE\xdf\xa3"  # EBML magic
    assert len(webm_two_seconds) > 1000


def test_decodes_stream_to_16khz_mono_pcm(webm_two_seconds):
    sink = ListSink()
    decoder = StreamingWebmDecoder(sink)

    for i in range(0, len(webm_two_seconds), 4096):
        decoder.feed(webm_two_seconds[i : i + 4096])
    decoder.close()

    assert decoder.error is None
    # 2 s at 16 kHz, allowing for opus priming/padding at the edges
    assert sink.samples == pytest.approx(2 * SAMPLE_RATE, rel=0.05)


def test_pcm_is_emitted_before_the_stream_is_closed(webm_two_seconds):
    """The decoder must not wait for EOF — this is what makes decoding incremental."""
    sink = ListSink()
    decoder = StreamingWebmDecoder(sink)
    try:
        prefix = webm_two_seconds[: int(len(webm_two_seconds) * 0.7)]
        for i in range(0, len(prefix), 4096):
            decoder.feed(prefix[i : i + 4096])

        assert wait_until(lambda: sink.samples > SAMPLE_RATE // 2), (
            f"no PCM before close (got {sink.samples} samples)"
        )
    finally:
        decoder.close()


def test_decoded_audio_preserves_the_original_tone(webm_two_seconds):
    """Byte counts alone would not catch a broken resample or wrong sample format."""
    sink = ListSink()
    decoder = StreamingWebmDecoder(sink)
    for i in range(0, len(webm_two_seconds), 4096):
        decoder.feed(webm_two_seconds[i : i + 4096])
    decoder.close()

    audio = sink.as_float()
    # ignore codec priming at the very start
    audio = audio[SAMPLE_RATE // 4 : SAMPLE_RATE // 4 + SAMPLE_RATE]
    spectrum = np.abs(np.fft.rfft(audio))
    dominant = np.fft.rfftfreq(len(audio), 1 / SAMPLE_RATE)[np.argmax(spectrum)]

    assert dominant == pytest.approx(440.0, abs=15.0)
    assert np.abs(audio).max() > 0.3  # not silence, not clipped to nothing


def test_garbage_input_surfaces_an_error_instead_of_failing_silently():
    sink = ListSink()
    decoder = StreamingWebmDecoder(sink)

    decoder.feed(b"this is not a webm container at all" * 100)
    decoder.close()

    assert decoder.error is not None
    assert sink.samples == 0


def test_feeding_after_close_does_not_raise(webm_two_seconds):
    sink = ListSink()
    decoder = StreamingWebmDecoder(sink)
    decoder.feed(webm_two_seconds[:4096])
    decoder.close()

    decoder.feed(webm_two_seconds[4096:8192])  # late chunk from a racing socket


def test_close_is_idempotent(webm_two_seconds):
    decoder = StreamingWebmDecoder(ListSink())
    decoder.feed(webm_two_seconds)
    decoder.close()
    decoder.close()


def test_close_returns_promptly_when_no_audio_was_ever_fed():
    """A meeting where capture starts and stops immediately must not hang the session."""
    decoder = StreamingWebmDecoder(ListSink())

    started = time.monotonic()
    decoder.close()

    assert time.monotonic() - started < 5.0
