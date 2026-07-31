"""SQLite persistence for sessions, transcripts and summaries.

Segments are written as they are recognised rather than at the end, so a crash part way
through a long meeting loses at most the current window instead of the whole transcript.
"""
from __future__ import annotations

import logging
import sqlite3
import threading
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from .incremental import Segment
from .models import Session
from .summarizer import Summary

logger = logging.getLogger(__name__)

_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    chunks              INTEGER NOT NULL DEFAULT 0,
    audio_bytes         INTEGER NOT NULL DEFAULT 0,
    started_at          TEXT,
    stopped_at          TEXT,
    status              TEXT,
    file                TEXT,
    transcript_segments INTEGER NOT NULL DEFAULT 0,
    language            TEXT,
    duration_sec        REAL NOT NULL DEFAULT 0,
    error               TEXT
);

CREATE TABLE IF NOT EXISTS segments (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    text       TEXT NOT NULL,
    start_sec  REAL NOT NULL,
    end_sec    REAL NOT NULL,
    speaker    TEXT,
    created_at REAL NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_segments_session ON segments(session_id, id);

CREATE TABLE IF NOT EXISTS summaries (
    session_id    TEXT PRIMARY KEY,
    content       TEXT,
    model         TEXT,
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    status        TEXT,
    error         TEXT
);
"""


class SessionStore:
    """Durable home for everything the API serves after a meeting ends."""

    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        # the transcription worker writes from its own thread, the API reads from the
        # event loop thread, so the connection is shared under an explicit lock
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._lock = threading.Lock()
        with self._lock:
            self._conn.executescript(_SCHEMA)
            self._conn.commit()

    def save_session(self, session: Session) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO sessions (session_id, chunks, audio_bytes, started_at,
                                      stopped_at, status, file, transcript_segments,
                                      language, duration_sec, error)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    chunks=excluded.chunks,
                    audio_bytes=excluded.audio_bytes,
                    started_at=excluded.started_at,
                    stopped_at=excluded.stopped_at,
                    status=excluded.status,
                    file=excluded.file,
                    transcript_segments=excluded.transcript_segments,
                    language=excluded.language,
                    duration_sec=excluded.duration_sec,
                    error=excluded.error
                """,
                (
                    session.session_id,
                    session.chunks,
                    session.audio_bytes,
                    session.started_at,
                    session.stopped_at,
                    session.status,
                    session.file,
                    session.transcript_segments,
                    session.language,
                    session.duration_sec,
                    session.error,
                ),
            )
            self._conn.commit()

    def add_segments(self, session_id: str, segments: Sequence[Segment]) -> None:
        if not segments:
            return
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO segments (session_id, text, start_sec, end_sec, speaker, created_at)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                [
                    (session_id, s.text, s.start, s.end, s.speaker, s.timestamp)
                    for s in segments
                ],
            )
            self._conn.commit()

    def save_summary(self, summary: Summary) -> None:
        with self._lock:
            self._conn.execute(
                """
                INSERT INTO summaries (session_id, content, model, input_tokens,
                                       output_tokens, status, error)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(session_id) DO UPDATE SET
                    content=excluded.content,
                    model=excluded.model,
                    input_tokens=excluded.input_tokens,
                    output_tokens=excluded.output_tokens,
                    status=excluded.status,
                    error=excluded.error
                """,
                (
                    summary.session_id,
                    summary.content,
                    summary.model,
                    summary.input_tokens,
                    summary.output_tokens,
                    summary.status,
                    summary.error,
                ),
            )
            self._conn.commit()

    def load_sessions(self) -> Dict[str, Session]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM sessions ORDER BY started_at"
            ).fetchall()
        return {row["session_id"]: _row_to_session(row) for row in rows}

    def load_segments(self, session_id: str) -> List[Segment]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT * FROM segments WHERE session_id = ? ORDER BY id", (session_id,)
            ).fetchall()
        return [
            Segment(
                text=row["text"],
                start=row["start_sec"],
                end=row["end_sec"],
                speaker=row["speaker"],
                timestamp=row["created_at"],
            )
            for row in rows
        ]

    def load_summary(self, session_id: str) -> Optional[Summary]:
        with self._lock:
            row = self._conn.execute(
                "SELECT * FROM summaries WHERE session_id = ?", (session_id,)
            ).fetchone()
        if row is None:
            return None
        return Summary(
            session_id=row["session_id"],
            content=row["content"] or "",
            model=row["model"] or "",
            input_tokens=row["input_tokens"],
            output_tokens=row["output_tokens"],
            status=row["status"] or "",
            error=row["error"],
        )

    def close(self) -> None:
        with self._lock:
            self._conn.close()


def _row_to_session(row: sqlite3.Row) -> Session:
    return Session(
        session_id=row["session_id"],
        chunks=row["chunks"],
        audio_bytes=row["audio_bytes"],
        started_at=row["started_at"],
        stopped_at=row["stopped_at"],
        status=row["status"],
        file=row["file"],
        transcript_segments=row["transcript_segments"],
        language=row["language"],
        duration_sec=row["duration_sec"],
        error=row["error"],
    )
