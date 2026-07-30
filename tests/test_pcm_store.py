"""PcmStore — append-only 16 kHz mono PCM backed by disk, with bounded-memory reads.

This is the component that makes 4+ hour sessions possible: audio accumulates on
disk, and transcription only ever pulls a small tail window into RAM.
"""
from __future__ import annotations

import numpy as np
import pytest

from backend.pcm_store import PcmStore

SAMPLE_RATE = 16000


def s16le(*samples: int) -> bytes:
    """Build raw little-endian int16 PCM from sample values."""
    return np.array(samples, dtype="<i2").tobytes()


@pytest.fixture
def store(tmp_path):
    s = PcmStore(tmp_path / "audio.pcm")
    yield s
    s.close()


def test_empty_store_has_no_samples(store):
    assert store.total_samples == 0
    assert store.duration == 0.0


def test_append_then_read_returns_normalized_float32(store):
    store.append(s16le(0, 16384, -16384, 32767))

    audio = store.read(0)

    assert audio.dtype == np.float32
    np.testing.assert_allclose(
        audio, [0.0, 0.5, -0.5, 32767 / 32768], rtol=0, atol=1e-6
    )


def test_total_samples_and_duration_track_appends(store):
    store.append(s16le(*range(SAMPLE_RATE)))  # exactly 1 second
    assert store.total_samples == SAMPLE_RATE
    assert store.duration == pytest.approx(1.0)

    store.append(s16le(*range(SAMPLE_RATE // 2)))  # + half a second
    assert store.total_samples == SAMPLE_RATE + SAMPLE_RATE // 2
    assert store.duration == pytest.approx(1.5)


def test_read_returns_only_the_requested_window(store):
    store.append(s16le(*range(1000)))

    audio = store.read(100, 150)

    assert len(audio) == 50
    np.testing.assert_allclose(audio[0], 100 / 32768, atol=1e-6)
    np.testing.assert_allclose(audio[-1], 149 / 32768, atol=1e-6)


def test_read_past_end_clamps_instead_of_failing(store):
    store.append(s16le(*range(10)))

    audio = store.read(5, 9999)

    assert len(audio) == 5


def test_read_beyond_total_returns_empty(store):
    store.append(s16le(*range(10)))

    assert len(store.read(10)) == 0
    assert len(store.read(50, 60)) == 0


def test_odd_trailing_byte_is_not_exposed_as_a_sample(store):
    """A network chunk can split an int16 across two appends — never emit half a sample."""
    store.append(s16le(1, 2) + b"\x03")  # 5 bytes = 2 whole samples + 1 dangling

    assert store.total_samples == 2

    store.append(b"\x00")  # completes the third sample

    assert store.total_samples == 3
    np.testing.assert_allclose(store.read(2, 3), [3 / 32768], atol=1e-6)


def test_reading_a_window_of_a_long_session_stays_bounded(store):
    """A 4-hour session must not pull 4 hours of audio into RAM to transcribe its tail."""
    one_minute = np.zeros(SAMPLE_RATE * 60, dtype="<i2").tobytes()
    for _ in range(60):  # 1 hour of audio on disk
        store.append(one_minute)

    assert store.duration == pytest.approx(3600.0)

    window = store.read(store.total_samples - SAMPLE_RATE * 30)

    assert len(window) == SAMPLE_RATE * 30
    assert window.nbytes == SAMPLE_RATE * 30 * 4  # float32 window only, not the hour


def test_appending_after_close_is_dropped_not_raised(tmp_path):
    """The decoder thread can still be draining when a session tears down."""
    store = PcmStore(tmp_path / "audio.pcm")
    store.append(s16le(1, 2))
    store.close()

    store.append(s16le(3, 4))  # late write from the decoder thread

    assert store.total_samples == 2


def test_close_is_idempotent(tmp_path):
    store = PcmStore(tmp_path / "audio.pcm")
    store.close()
    store.close()


def test_data_survives_reopening_the_same_path(tmp_path):
    path = tmp_path / "audio.pcm"
    first = PcmStore(path)
    first.append(s16le(1, 2, 3))
    first.close()

    reopened = PcmStore(path)
    try:
        assert reopened.total_samples == 3
        np.testing.assert_allclose(reopened.read(0, 1), [1 / 32768], atol=1e-6)
    finally:
        reopened.close()
