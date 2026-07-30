"""Whisper model reuse and the per-session transcribe adapter.

Two things matter here:
  1. The model is loaded once per process. Loading it per session leaked a full model
     into RAM for every meeting and made each meeting start cold.
  2. Language is a per-call argument, so one shared model can serve sessions in
     different languages — which is what makes a per-session language picker possible.
"""
from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from backend.transcriber import WhisperTranscriber, get_model, reset_model_cache


class FakeModel:
    def __init__(self, tag: str = "model"):
        self.tag = tag
        self.calls: list[dict] = []

    def transcribe(self, audio, **kwargs):
        self.calls.append({"samples": len(audio), **kwargs})

        def generate():
            yield SimpleNamespace(start=0.0, end=1.0, text="hello")
            yield SimpleNamespace(start=1.0, end=2.0, text="world")

        return generate(), SimpleNamespace(language="en", language_probability=0.99)


class RecordingFactory:
    def __init__(self):
        self.builds: list[tuple] = []

    def __call__(self, model_size, device=None, compute_type=None):
        self.builds.append((model_size, device, compute_type))
        return FakeModel(tag=f"{model_size}/{device}/{compute_type}")


@pytest.fixture(autouse=True)
def clean_cache():
    reset_model_cache()
    yield
    reset_model_cache()


def test_same_config_reuses_one_loaded_model():
    factory = RecordingFactory()

    first = get_model("base", "cpu", "float32", factory=factory)
    second = get_model("base", "cpu", "float32", factory=factory)

    assert first is second
    assert len(factory.builds) == 1


def test_different_config_loads_a_separate_model():
    factory = RecordingFactory()

    base = get_model("base", "cpu", "float32", factory=factory)
    small = get_model("small", "cpu", "float32", factory=factory)

    assert base is not small
    assert len(factory.builds) == 2


def test_many_sessions_share_a_single_model():
    """Regression: a model used to be loaded per WebSocket connection and never freed."""
    factory = RecordingFactory()

    models = [get_model("base", "cpu", "float32", factory=factory) for _ in range(50)]

    assert len({id(m) for m in models}) == 1
    assert len(factory.builds) == 1


def test_reset_clears_the_cache():
    factory = RecordingFactory()

    get_model("base", "cpu", "float32", factory=factory)
    reset_model_cache()
    get_model("base", "cpu", "float32", factory=factory)

    assert len(factory.builds) == 2


def test_adapter_returns_a_concrete_list_not_a_lazy_generator():
    """faster-whisper defers all inference into the generator — it must be consumed."""
    model = FakeModel()
    transcribe = WhisperTranscriber(model)

    result = transcribe(np.zeros(16000, dtype=np.float32))

    assert isinstance(result, list)
    assert [s.text for s in result] == ["hello", "world"]


def test_adapter_passes_the_session_language_through():
    model = FakeModel()

    WhisperTranscriber(model, language="ko")(np.zeros(16000, dtype=np.float32))

    assert model.calls[0]["language"] == "ko"


def test_adapter_leaves_language_unset_for_auto_detect():
    model = FakeModel()

    WhisperTranscriber(model)(np.zeros(16000, dtype=np.float32))

    assert model.calls[0]["language"] is None


def test_two_sessions_can_use_different_languages_on_one_model():
    """Non-English support was the differentiator called out on the thread."""
    model = FakeModel()

    WhisperTranscriber(model, language="ko")(np.zeros(16000, dtype=np.float32))
    WhisperTranscriber(model, language="ja")(np.zeros(16000, dtype=np.float32))

    assert [c["language"] for c in model.calls] == ["ko", "ja"]


def test_adapter_enables_voice_activity_filtering():
    model = FakeModel()

    WhisperTranscriber(model)(np.zeros(16000, dtype=np.float32))

    assert model.calls[0]["vad_filter"] is True
