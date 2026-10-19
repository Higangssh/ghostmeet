//! SQLite persistence for sessions, transcripts and summaries.
//!
//! Segments are written as they are committed rather than at the end, so a crash part way
//! through a long meeting loses at most the transcript of the newest window instead of
//! the whole transcript: up to `window_sec` of speech held back or not yet committed,
//! plus the audio that arrived since the last pass. The audio itself is already on disk
//! (`PcmStore`).
//!
//! After the meeting, the final transcript replaces the live one in one transaction, so
//! a reader sees one or the other and a failure leaves the live transcript as it was.

use std::path::Path;
use std::sync::{Mutex, MutexGuard};

use rusqlite::{params, Connection, OptionalExtension, Row};
use serde::{Deserialize, Serialize};

use crate::incremental::Segment;

pub use rusqlite::Error as StoreError;
pub type Result<T> = std::result::Result<T, StoreError>;

const SCHEMA_VERSION: i64 = 1;

const SCHEMA: &str = "
CREATE TABLE IF NOT EXISTS sessions (
    session_id          TEXT PRIMARY KEY,
    title               TEXT NOT NULL DEFAULT '',
    started_at          REAL NOT NULL,
    stopped_at          REAL,
    status              TEXT NOT NULL,
    sources             TEXT NOT NULL DEFAULT '',
    language            TEXT,
    duration_sec        REAL NOT NULL DEFAULT 0,
    transcript          TEXT NOT NULL DEFAULT 'live',
    transcript_segments INTEGER NOT NULL DEFAULT 0,
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
    content       TEXT NOT NULL DEFAULT '',
    model         TEXT NOT NULL DEFAULT '',
    input_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens INTEGER NOT NULL DEFAULT 0,
    status        TEXT NOT NULL DEFAULT '',
    error         TEXT
);
";

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum SessionStatus {
    /// Capturing audio, live captions running.
    Recording,
    /// Capture stopped; the live pass is draining what is left.
    Transcribing,
    /// The after-meeting pass is producing the final transcript.
    Finalizing,
    Stopped,
    /// Left mid-flight by a process that is gone.
    Interrupted,
    Error,
}

impl SessionStatus {
    fn as_str(self) -> &'static str {
        match self {
            Self::Recording => "recording",
            Self::Transcribing => "transcribing",
            Self::Finalizing => "finalizing",
            Self::Stopped => "stopped",
            Self::Interrupted => "interrupted",
            Self::Error => "error",
        }
    }

    fn parse(s: &str) -> Self {
        match s {
            "recording" => Self::Recording,
            "transcribing" => Self::Transcribing,
            "finalizing" => Self::Finalizing,
            "stopped" => Self::Stopped,
            "interrupted" => Self::Interrupted,
            _ => Self::Error,
        }
    }
}

/// Which pass the stored transcript came from.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "lowercase")]
pub enum TranscriptKind {
    /// Provisional captions from the small model, written during the meeting.
    Live,
    /// The after-meeting pass with the large model.
    Final,
}

impl TranscriptKind {
    fn as_str(self) -> &'static str {
        match self {
            Self::Live => "live",
            Self::Final => "final",
        }
    }

    fn parse(s: &str) -> Self {
        if s == "final" {
            Self::Final
        } else {
            Self::Live
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Session {
    pub session_id: String,
    pub title: String,
    /// Wall-clock seconds since the Unix epoch.
    pub started_at: f64,
    pub stopped_at: Option<f64>,
    pub status: SessionStatus,
    /// Audio sources captured, e.g. `["mic", "system"]`.
    pub sources: Vec<String>,
    pub language: Option<String>,
    pub duration_sec: f64,
    pub transcript: TranscriptKind,
    pub transcript_segments: i64,
    pub error: Option<String>,
}

impl Session {
    pub fn new(session_id: impl Into<String>, started_at: f64) -> Self {
        Self {
            session_id: session_id.into(),
            title: String::new(),
            started_at,
            stopped_at: None,
            status: SessionStatus::Recording,
            sources: Vec::new(),
            language: None,
            duration_sec: 0.0,
            transcript: TranscriptKind::Live,
            transcript_segments: 0,
            error: None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct Summary {
    pub session_id: String,
    pub content: String,
    pub model: String,
    pub input_tokens: i64,
    pub output_tokens: i64,
    pub status: String,
    pub error: Option<String>,
}

impl Summary {
    pub fn new(session_id: impl Into<String>) -> Self {
        Self {
            session_id: session_id.into(),
            content: String::new(),
            model: String::new(),
            input_tokens: 0,
            output_tokens: 0,
            status: "done".into(),
            error: None,
        }
    }
}

/// Durable home for every meeting.
///
/// One connection shared under a lock: each channel's transcription worker writes, the
/// UI reads.
pub struct SessionStore {
    conn: Mutex<Connection>,
}

impl SessionStore {
    pub fn open(path: impl AsRef<Path>) -> Result<Self> {
        let path = path.as_ref();
        if let Some(parent) = path.parent() {
            std::fs::create_dir_all(parent)
                .map_err(|e| StoreError::ToSqlConversionFailure(Box::new(e)))?;
        }
        let conn = Connection::open(path)?;
        conn.pragma_update(None, "journal_mode", "WAL")?;
        conn.execute_batch(SCHEMA)?;
        conn.pragma_update(None, "user_version", SCHEMA_VERSION)?;
        Ok(Self {
            conn: Mutex::new(conn),
        })
    }

    fn conn(&self) -> MutexGuard<'_, Connection> {
        // a panicked writer cannot leave a half-applied statement behind: SQLite rolls
        // back anything not committed, so the connection is still sound
        self.conn.lock().unwrap_or_else(|e| e.into_inner())
    }

    pub fn save_session(&self, s: &Session) -> Result<()> {
        self.conn().execute(
            "INSERT INTO sessions (session_id, title, started_at, stopped_at, status, sources,
                                   language, duration_sec, transcript, transcript_segments, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)
             ON CONFLICT(session_id) DO UPDATE SET
                 title=excluded.title,
                 started_at=excluded.started_at,
                 stopped_at=excluded.stopped_at,
                 status=excluded.status,
                 sources=excluded.sources,
                 language=excluded.language,
                 duration_sec=excluded.duration_sec,
                 transcript=excluded.transcript,
                 transcript_segments=excluded.transcript_segments,
                 error=excluded.error",
            params![
                s.session_id,
                s.title,
                s.started_at,
                s.stopped_at,
                s.status.as_str(),
                s.sources.join(","),
                s.language,
                s.duration_sec,
                s.transcript.as_str(),
                s.transcript_segments,
                s.error,
            ],
        )?;
        Ok(())
    }

    pub fn load_sessions(&self) -> Result<Vec<Session>> {
        let conn = self.conn();
        let mut stmt = conn.prepare("SELECT * FROM sessions ORDER BY started_at, session_id")?;
        let rows = stmt.query_map([], row_to_session)?;
        rows.collect()
    }

    pub fn load_session(&self, session_id: &str) -> Result<Option<Session>> {
        self.conn()
            .query_row(
                "SELECT * FROM sessions WHERE session_id = ?1",
                [session_id],
                row_to_session,
            )
            .optional()
    }

    pub fn rename_session(&self, session_id: &str, title: &str) -> Result<()> {
        self.conn().execute(
            "UPDATE sessions SET title = ?2 WHERE session_id = ?1",
            params![session_id, title],
        )?;
        Ok(())
    }

    /// Remove a session with its transcript and summary. Its audio is the caller's.
    pub fn delete_session(&self, session_id: &str) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        for table in ["segments", "summaries", "sessions"] {
            tx.execute(
                &format!("DELETE FROM {table} WHERE session_id = ?1"),
                [session_id],
            )?;
        }
        tx.commit()
    }

    /// Sessions a dead process left mid-flight become `Interrupted`, transcript kept.
    /// Call once at startup, before anything starts capturing.
    pub fn recover_interrupted(&self) -> Result<usize> {
        self.conn().execute(
            "UPDATE sessions SET status = 'interrupted'
             WHERE status IN ('recording', 'transcribing', 'finalizing')",
            [],
        )
    }

    /// Append live segments as they are recognised.
    pub fn add_segments(&self, session_id: &str, segments: &[Segment]) -> Result<()> {
        if segments.is_empty() {
            return Ok(());
        }
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        insert_segments(&tx, session_id, segments)?;
        tx.commit()
    }

    /// Swap the whole transcript for the final one, all or nothing.
    pub fn replace_transcript(&self, session_id: &str, segments: &[Segment]) -> Result<()> {
        let mut conn = self.conn();
        let tx = conn.transaction()?;
        let updated = tx.execute(
            "UPDATE sessions SET transcript = 'final', transcript_segments = ?2
             WHERE session_id = ?1",
            params![session_id, segments.len() as i64],
        )?;
        if updated == 0 {
            return Err(StoreError::QueryReturnedNoRows);
        }
        tx.execute("DELETE FROM segments WHERE session_id = ?1", [session_id])?;
        insert_segments(&tx, session_id, segments)?;
        tx.commit()
    }

    pub fn load_segments(&self, session_id: &str) -> Result<Vec<Segment>> {
        let conn = self.conn();
        let mut stmt = conn.prepare(
            "SELECT text, start_sec, end_sec, speaker, created_at FROM segments
             WHERE session_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map([session_id], |r| {
            Ok(Segment {
                text: r.get(0)?,
                start: r.get(1)?,
                end: r.get(2)?,
                speaker: r.get(3)?,
                timestamp: r.get(4)?,
            })
        })?;
        rows.collect()
    }

    pub fn save_summary(&self, s: &Summary) -> Result<()> {
        self.conn().execute(
            "INSERT INTO summaries (session_id, content, model, input_tokens, output_tokens,
                                    status, error)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)
             ON CONFLICT(session_id) DO UPDATE SET
                 content=excluded.content,
                 model=excluded.model,
                 input_tokens=excluded.input_tokens,
                 output_tokens=excluded.output_tokens,
                 status=excluded.status,
                 error=excluded.error",
            params![
                s.session_id,
                s.content,
                s.model,
                s.input_tokens,
                s.output_tokens,
                s.status,
                s.error
            ],
        )?;
        Ok(())
    }

    pub fn load_summary(&self, session_id: &str) -> Result<Option<Summary>> {
        self.conn()
            .query_row(
                "SELECT * FROM summaries WHERE session_id = ?1",
                [session_id],
                |r| {
                    Ok(Summary {
                        session_id: r.get("session_id")?,
                        content: r.get("content")?,
                        model: r.get("model")?,
                        input_tokens: r.get("input_tokens")?,
                        output_tokens: r.get("output_tokens")?,
                        status: r.get("status")?,
                        error: r.get("error")?,
                    })
                },
            )
            .optional()
    }
}

fn insert_segments(
    tx: &rusqlite::Transaction,
    session_id: &str,
    segments: &[Segment],
) -> Result<()> {
    let mut stmt = tx.prepare(
        "INSERT INTO segments (session_id, text, start_sec, end_sec, speaker, created_at)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
    )?;
    for s in segments {
        stmt.execute(params![
            session_id,
            s.text,
            s.start,
            s.end,
            s.speaker,
            s.timestamp
        ])?;
    }
    Ok(())
}

fn row_to_session(r: &Row) -> rusqlite::Result<Session> {
    let sources: String = r.get("sources")?;
    Ok(Session {
        session_id: r.get("session_id")?,
        title: r.get("title")?,
        started_at: r.get("started_at")?,
        stopped_at: r.get("stopped_at")?,
        status: SessionStatus::parse(&r.get::<_, String>("status")?),
        sources: sources
            .split(',')
            .filter(|s| !s.is_empty())
            .map(str::to_string)
            .collect(),
        language: r.get("language")?,
        duration_sec: r.get("duration_sec")?,
        transcript: TranscriptKind::parse(&r.get::<_, String>("transcript")?),
        transcript_segments: r.get("transcript_segments")?,
        error: r.get("error")?,
    })
}
