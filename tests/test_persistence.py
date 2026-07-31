"""Transcripts and summaries must outlive the backend process.

Entering the TestClient context runs the app's startup, so leaving and re-entering it
is a genuine restart: in-memory state is dropped and everything has to come back from
disk. Without this the OpenClaw skill cannot answer "summarise my last meeting".
"""
from __future__ import annotations

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


def stub_factory(language=None):
    def transcribe(audio):
        return [RawSeg(0.0, 1.0, "we agreed to ship on friday")]

    return transcribe


@pytest.fixture
def wired(monkeypatch, tmp_path):
    """Point the app at a temp recordings dir and database."""
    monkeypatch.setattr(app_module, "_make_transcribe_fn", stub_factory)
    monkeypatch.setattr(app_module, "RECORDINGS_DIR", tmp_path)
    monkeypatch.setattr(app_module, "DB_PATH", tmp_path / "ghostmeet.db")
    return tmp_path


@pytest.fixture(scope="module")
def webm() -> bytes:
    return make_webm_opus(seconds=2.0)


def stream(client, webm: bytes, session_id: str) -> None:
    with client.websocket_connect(f"/ws/audio?session={session_id}") as ws:
        ws.receive_json()
        for i in range(0, len(webm), 4096):
            ws.send_bytes(webm[i : i + 4096])
        ws.send_text("stop")
        assert ws.receive_json()["type"] == "complete"


def test_transcript_survives_a_restart(wired, webm):
    with TestClient(app_module.app) as client:
        stream(client, webm, "persisted")
        before = client.get("/api/sessions/persisted/transcript").json()

    with TestClient(app_module.app) as client:
        after = client.get("/api/sessions/persisted/transcript").json()

    assert before["segment_count"] >= 1
    assert after["segment_count"] == before["segment_count"]
    assert after["full_text"] == before["full_text"]
    assert "ship on friday" in after["full_text"]


def test_session_list_survives_a_restart(wired, webm):
    with TestClient(app_module.app) as client:
        stream(client, webm, "persisted")

    with TestClient(app_module.app) as client:
        body = client.get("/api/sessions").json()

    assert "persisted" in body["sessions"]
    session = body["sessions"]["persisted"]
    assert session["status"] == "stopped"
    assert session["audio_bytes"] == len(webm)
    assert session["duration_sec"] > 1.0


def test_segments_are_stored_with_their_speaker_slot(wired, webm):
    with TestClient(app_module.app) as client:
        stream(client, webm, "persisted")

    with TestClient(app_module.app) as client:
        segments = client.get("/api/sessions/persisted/transcript").json()["segments"]

    assert segments[0]["speaker"] is None
    assert segments[0]["end"] > segments[0]["start"]


def test_a_meeting_can_be_summarised_after_a_restart(wired, webm, monkeypatch):
    """This is the flow the OpenClaw skill promises and could not deliver."""
    with TestClient(app_module.app) as client:
        stream(client, webm, "persisted")

    captured = {}

    async def fake_summary(text, session_id, *args, **kwargs):
        from backend.summarizer import Summary

        captured["text"] = text
        return Summary(session_id=session_id, content="- ship on friday", status="done")

    monkeypatch.setattr(app_module, "generate_summary", fake_summary)

    with TestClient(app_module.app) as client:
        response = client.post("/api/sessions/persisted/summarize")

    assert response.status_code == 200
    assert "ship on friday" in captured["text"]


def test_summary_survives_a_restart(wired, webm, monkeypatch):
    async def fake_summary(text, session_id, *args, **kwargs):
        from backend.summarizer import Summary

        return Summary(session_id=session_id, content="decisions: ship", status="done")

    monkeypatch.setattr(app_module, "generate_summary", fake_summary)

    with TestClient(app_module.app) as client:
        stream(client, webm, "persisted")
        client.post("/api/sessions/persisted/summarize")

    with TestClient(app_module.app) as client:
        body = client.get("/api/sessions/persisted/summary").json()

    assert body["content"] == "decisions: ship"
    assert body["status"] == "done"


def test_segments_are_written_during_the_meeting_not_only_at_the_end(wired, webm, monkeypatch):
    """A crash four hours in should not cost the whole transcript."""
    monkeypatch.setattr(app_module, "CHUNK_INTERVAL", 0.05)

    with TestClient(app_module.app) as client:
        with client.websocket_connect("/ws/audio?session=live") as ws:
            ws.receive_json()
            for i in range(0, len(webm), 4096):
                ws.send_bytes(webm[i : i + 4096])

            stored = []
            for _ in range(100):
                stored = app_module.store.load_segments("live")
                if stored:
                    break
                client.get("/api/health")

            assert stored, "nothing was persisted while the session was still open"
            ws.send_text("stop")
            ws.receive_json()


def test_unknown_session_is_still_a_404_after_a_restart(wired):
    with TestClient(app_module.app) as client:
        assert client.get("/api/sessions/never-existed/transcript").status_code == 404
        assert client.get("/api/sessions/never-existed/summary").status_code == 404


def test_a_live_session_is_served_from_memory_not_the_database(wired, webm, monkeypatch):
    """While a meeting runs, captions come from the pipeline that is producing them."""
    monkeypatch.setattr(app_module, "CHUNK_INTERVAL", 0.05)

    with TestClient(app_module.app) as client:
        with client.websocket_connect("/ws/audio?session=live") as ws:
            ws.receive_json()
            for i in range(0, len(webm), 4096):
                ws.send_bytes(webm[i : i + 4096])

            for _ in range(100):
                body = client.get("/api/sessions/live/transcript").json()
                if body["segment_count"]:
                    break

            assert body["segment_count"] >= 1
            ws.send_text("stop")
            ws.receive_json()
