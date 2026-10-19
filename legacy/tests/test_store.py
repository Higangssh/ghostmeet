"""SessionStore — transcripts that outlive the process.

Everything used to live in module-level dicts, so restarting the backend threw away
every transcript and summary. That also broke the OpenClaw skill's whole premise of
asking about a meeting after the fact.
"""
from __future__ import annotations

import threading

import pytest

from backend.incremental import Segment
from backend.models import Session
from backend.store import SessionStore
from backend.summarizer import Summary


@pytest.fixture
def store(tmp_path):
    s = SessionStore(tmp_path / "ghostmeet.db")
    yield s
    s.close()


def make_session(session_id="20260731-090000", **overrides) -> Session:
    session = Session(session_id=session_id, file=f"{session_id}.webm")
    session.chunks = 12
    session.audio_bytes = 40960
    session.status = "stopped"
    session.stopped_at = "2026-07-31T09:412:00"
    session.transcript_segments = 3
    session.language = "ko"
    session.duration_sec = 128.5
    for key, value in overrides.items():
        setattr(session, key, value)
    return session


def test_a_fresh_store_has_no_sessions(store):
    assert store.load_sessions() == {}


def test_session_round_trips_every_field(store):
    original = make_session()

    store.save_session(original)
    loaded = store.load_sessions()["20260731-090000"]

    assert loaded.to_dict() == original.to_dict()


def test_saving_the_same_session_updates_it(store):
    session = make_session()
    store.save_session(session)

    session.status = "stopped"
    session.transcript_segments = 99
    store.save_session(session)

    sessions = store.load_sessions()
    assert len(sessions) == 1
    assert sessions["20260731-090000"].transcript_segments == 99


def test_segments_come_back_in_the_order_they_were_added(store):
    store.save_session(make_session("s1"))
    store.add_segments("s1", [Segment(text="first", start=0.0, end=1.0)])
    store.add_segments(
        "s1",
        [Segment(text="second", start=1.0, end=2.0), Segment(text="third", start=2.0, end=3.0)],
    )

    assert [s.text for s in store.load_segments("s1")] == ["first", "second", "third"]


def test_segment_fields_survive_including_speaker(store):
    store.save_session(make_session("s1"))
    store.add_segments(
        "s1", [Segment(text="hello", start=1.5, end=2.25, speaker="A", timestamp=1234.5)]
    )

    loaded = store.load_segments("s1")[0]

    assert loaded.text == "hello"
    assert loaded.start == pytest.approx(1.5)
    assert loaded.end == pytest.approx(2.25)
    assert loaded.speaker == "A"
    assert loaded.timestamp == pytest.approx(1234.5)


def test_segments_do_not_leak_between_sessions(store):
    store.save_session(make_session("s1"))
    store.save_session(make_session("s2"))
    store.add_segments("s1", [Segment(text="mine", start=0.0, end=1.0)])
    store.add_segments("s2", [Segment(text="yours", start=0.0, end=1.0)])

    assert [s.text for s in store.load_segments("s1")] == ["mine"]
    assert [s.text for s in store.load_segments("s2")] == ["yours"]


def test_unknown_session_has_no_segments(store):
    assert store.load_segments("never-happened") == []


def test_adding_no_segments_is_harmless(store):
    store.save_session(make_session("s1"))

    store.add_segments("s1", [])

    assert store.load_segments("s1") == []


def test_summary_round_trips(store):
    store.save_session(make_session("s1"))
    summary = Summary(
        session_id="s1",
        content="## Decisions\n- ship it",
        model="claude-sonnet-4",
        input_tokens=120,
        output_tokens=45,
        status="done",
    )

    store.save_summary(summary)

    assert store.load_summary("s1").to_dict() == summary.to_dict()


def test_regenerating_a_summary_replaces_the_old_one(store):
    store.save_session(make_session("s1"))
    store.save_summary(Summary(session_id="s1", content="first pass", status="done"))
    store.save_summary(Summary(session_id="s1", content="second pass", status="done"))

    assert store.load_summary("s1").content == "second pass"


def test_missing_summary_reads_as_none(store):
    assert store.load_summary("s1") is None


def test_everything_survives_reopening_the_database(tmp_path):
    """The actual point: restarting the backend must not lose a meeting."""
    path = tmp_path / "ghostmeet.db"
    first = SessionStore(path)
    first.save_session(make_session("s1"))
    first.add_segments("s1", [Segment(text="we decided to ship", start=0.0, end=2.0)])
    first.save_summary(Summary(session_id="s1", content="shipped", status="done"))
    first.close()

    reopened = SessionStore(path)
    try:
        assert "s1" in reopened.load_sessions()
        assert [s.text for s in reopened.load_segments("s1")] == ["we decided to ship"]
        assert reopened.load_summary("s1").content == "shipped"
    finally:
        reopened.close()


def test_writes_from_several_threads_all_land(store):
    """Segments are written from the transcription worker, not the event loop thread."""
    store.save_session(make_session("s1"))

    def write(index: int):
        store.add_segments(
            "s1", [Segment(text=f"line {index}", start=float(index), end=index + 1.0)]
        )

    threads = [threading.Thread(target=write, args=(i,)) for i in range(20)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert len(store.load_segments("s1")) == 20


def test_reopening_an_existing_database_does_not_wipe_it(tmp_path):
    path = tmp_path / "ghostmeet.db"
    first = SessionStore(path)
    first.save_session(make_session("s1"))
    first.close()

    second = SessionStore(path)
    second.save_session(make_session("s2"))
    try:
        assert set(second.load_sessions()) == {"s1", "s2"}
    finally:
        second.close()
