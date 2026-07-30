"""End-to-end: real webm over the WebSocket, out the other side as a transcript.

Uses a stub transcribe function so the suite never downloads a Whisper model, but the
audio path is real — actual opus bytes through the actual decoder.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import pytest
from fastapi.testclient import TestClient

from backend import app as app_module
from .test_decoder import make_webm_opus


@dataclass
class RawSeg:
    start: float
    end: float
    text: str


class StubWhisper:
    """Records the audio it was given and the language it was built with."""

    def __init__(self, language=None):
        self.language = language
        self.samples_seen = 0

    def __call__(self, audio):
        self.samples_seen += len(audio)
        return [RawSeg(0.0, 1.0, "transcribed text")]


@pytest.fixture
def built():
    """Track every transcribe function the app builds, and the languages requested."""
    made: list[StubWhisper] = []

    def factory(language=None):
        stub = StubWhisper(language=language)
        made.append(stub)
        return stub

    return made, factory


@pytest.fixture
def client(monkeypatch, tmp_path, built):
    made, factory = built
    monkeypatch.setattr(app_module, "_make_transcribe_fn", factory)
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(app_module, "sessions", {})
    monkeypatch.setattr(app_module, "pipelines", {})
    monkeypatch.setattr(app_module, "summaries", {})
    with TestClient(app_module.app) as c:
        yield c


@pytest.fixture(scope="module")
def webm() -> bytes:
    return make_webm_opus(seconds=2.0)


def stream(client, webm: bytes, query: str = "") -> str:
    """Push a whole webm through /ws/audio, wait for the final pass, return session id."""
    with client.websocket_connect(f"/ws/audio{query}") as ws:
        session_id = ws.receive_json()["session_id"]
        for i in range(0, len(webm), 4096):
            ws.send_bytes(webm[i : i + 4096])
        ws.send_text("stop")
        done = ws.receive_json()
        assert done["type"] == "complete"
    return session_id


def test_health_reports_the_active_configuration(client):
    body = client.get("/api/health").json()

    assert body["ok"] is True
    assert "model" in body
    assert "chunk_interval_sec" in body


def test_audio_stream_produces_a_transcript(client, webm):
    session_id = stream(client, webm)

    body = client.get(f"/api/sessions/{session_id}/transcript").json()

    assert body["segment_count"] >= 1
    assert "transcribed text" in body["full_text"]
    assert body["segments"][0]["start"] == pytest.approx(0.0, abs=0.01)


def test_real_audio_actually_reaches_the_transcriber(client, webm, built):
    made, _ = built

    stream(client, webm)

    assert len(made) == 1
    assert made[0].samples_seen > 16000  # roughly a second or more of decoded audio


def test_session_is_listed_with_capture_stats(client, webm):
    session_id = stream(client, webm)

    body = client.get("/api/sessions").json()

    assert session_id in body["sessions"]
    session = body["sessions"][session_id]
    assert session["status"] == "stopped"
    assert session["audio_bytes"] == len(webm)
    assert session["chunks"] > 1
    assert session["transcript_segments"] >= 1


def test_requested_session_id_is_honoured(client, webm):
    session_id = stream(client, webm, query="?session=my-standup")

    assert session_id == "my-standup"
    assert client.get("/api/sessions/my-standup").status_code == 200


def test_language_can_be_chosen_per_session(client, webm, built):
    """Non-English accuracy was the differentiator raised on the thread."""
    made, _ = built

    stream(client, webm, query="?session=a&lang=ko")
    stream(client, webm, query="?session=b&lang=ja")

    assert [stub.language for stub in made] == ["ko", "ja"]


def test_language_defaults_to_auto_detect(client, webm, built):
    made, _ = built

    stream(client, webm)

    assert made[0].language is None


def test_segments_include_a_speaker_field(client, webm):
    session_id = stream(client, webm)

    segments = client.get(f"/api/sessions/{session_id}/transcript").json()["segments"]

    assert "speaker" in segments[0]


def test_reusing_a_finished_session_id_starts_fresh(client, webm, tmp_path):
    """Regression: the old handler appended to the previous recording, corrupting both
    the archive and the transcript."""
    stream(client, webm, query="?session=repeat")
    first = client.get("/api/sessions/repeat").json()

    stream(client, webm, query="?session=repeat")
    second = client.get("/api/sessions/repeat").json()

    assert first["audio_bytes"] == len(webm)
    assert second["audio_bytes"] == len(webm)  # not doubled
    assert (tmp_path / "repeat.webm").stat().st_size == len(webm)


def test_the_pcm_working_file_is_reclaimed_but_the_archive_is_kept(client, webm, tmp_path):
    """A 4-hour session leaves ~450 MB of working PCM behind if this is not cleaned up."""
    session_id = stream(client, webm)

    assert not (tmp_path / f"{session_id}.pcm").exists()
    assert (tmp_path / f"{session_id}.webm").exists()


def test_completion_notice_reports_the_final_state(client, webm):
    with client.websocket_connect("/ws/audio?session=notice") as ws:
        ws.receive_json()
        for i in range(0, len(webm), 4096):
            ws.send_bytes(webm[i : i + 4096])
        ws.send_text("stop")
        done = ws.receive_json()

    assert done["session_id"] == "notice"
    assert done["status"] == "stopped"
    assert done["segment_count"] >= 1
    assert done["duration_sec"] > 1.0


def test_hanging_up_without_stop_does_not_leak_the_working_file(client, webm, tmp_path):
    """Closing the tab mid-meeting must still release handles and reclaim disk.

    Whether the final transcription pass completes on an abrupt hangup depends on the
    server not cancelling the handler task; that is covered at the pipeline level and
    becomes moot once transcripts are persisted.
    """
    with client.websocket_connect("/ws/audio?session=dropped") as ws:
        ws.receive_json()
        for i in range(0, len(webm), 4096):
            ws.send_bytes(webm[i : i + 4096])
        # no "stop" — just hang up

    for _ in range(100):
        if not (tmp_path / "dropped.pcm").exists():
            break
        time.sleep(0.05)

    assert not (tmp_path / "dropped.pcm").exists()
    assert (tmp_path / "dropped.webm").exists()


def test_unknown_session_returns_404(client):
    assert client.get("/api/sessions/nope").status_code == 404
    assert client.get("/api/sessions/nope/transcript").status_code == 404


def test_summarize_rejects_an_empty_transcript(client, webm, monkeypatch):
    def silent(language=None):
        return lambda audio: []

    monkeypatch.setattr(app_module, "_make_transcribe_fn", silent)
    session_id = stream(client, webm)

    response = client.post(f"/api/sessions/{session_id}/summarize")

    assert response.status_code == 400


def test_live_subscriber_receives_captions_during_the_session(client, webm, monkeypatch):
    monkeypatch.setattr(app_module, "CHUNK_INTERVAL", 0.05)

    with client.websocket_connect("/ws/transcript/live-test") as sub:
        with client.websocket_connect("/ws/audio?session=live-test") as ws:
            ws.receive_json()
            for i in range(0, len(webm), 4096):
                ws.send_bytes(webm[i : i + 4096])
            message = sub.receive_json()
            ws.send_text("stop")
            assert ws.receive_json()["type"] == "complete"

    assert message["type"] == "transcript"
    assert message["segments"][0]["text"] == "transcribed text"


def test_sessions_go_through_the_shared_model_cache(monkeypatch):
    """Regression: the model used to be loaded per connection and never released."""
    from backend import transcriber as transcriber_module

    builds = []

    def counting_load(model_size, device, compute_type):
        builds.append(model_size)
        return object()

    monkeypatch.setattr(transcriber_module, "_load_whisper", counting_load)
    transcriber_module.reset_model_cache()
    try:
        app_module._make_transcribe_fn("ko")
        app_module._make_transcribe_fn("ja")

        assert len(builds) == 1
    finally:
        transcriber_module.reset_model_cache()


def test_transcribe_factory_applies_the_requested_language(monkeypatch):
    from backend import transcriber as transcriber_module

    monkeypatch.setattr(
        transcriber_module, "_load_whisper", lambda *a, **k: object()
    )
    transcriber_module.reset_model_cache()
    try:
        assert app_module._make_transcribe_fn("ko").language == "ko"
        assert app_module._make_transcribe_fn(None).language is None
    finally:
        transcriber_module.reset_model_cache()
