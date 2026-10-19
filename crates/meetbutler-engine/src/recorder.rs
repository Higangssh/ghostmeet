//! A meeting from start to stop: one PcmStore and one transcription pipeline per
//! channel, segments persisted as they are recognised.
//!
//! Layout of the data folder (one folder, so backing up or deleting is easy):
//!
//! ```text
//! meetbutler.db
//! sessions/<session_id>/mic.pcm      16 kHz mono s16le, the microphone ("me")
//! sessions/<session_id>/system.pcm   the same for system audio ("them")
//! ```

use std::collections::HashMap;
use std::io;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard};
use std::time::{Duration, SystemTime, UNIX_EPOCH};

use crate::incremental::{IncrementalTranscriber, Segment, WindowConfig};
use crate::pcm_store::PcmStore;
use crate::pipeline::SessionPipeline;
use crate::span_decoder::{final_transcript, SpanDecode, SpeechProbs};
use crate::spans::BENCH;
use crate::store::{Session, SessionStatus, SessionStore, StoreError, TranscriptKind};
use crate::{Transcribe, TranscribeError};

/// The after-meeting pass cuts a channel this much at a time.
const FINAL_CHUNK_SEC: f64 = 600.0;

/// Called with every batch of new segments, after they are on disk.
pub type OnSegments = Arc<dyn Fn(&str, &[Segment]) + Send + Sync>;

#[derive(Clone)]
pub struct RecorderConfig {
    /// How often a channel runs a live pass.
    pub interval: Duration,
    pub window: WindowConfig,
    pub on_segments: Option<OnSegments>,
}

impl Default for RecorderConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(3),
            window: WindowConfig::default(),
            on_segments: None,
        }
    }
}

/// Where a channel's audio comes from. Until diarization, it is also who spoke.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Channel {
    /// The microphone: the user, and in a room everyone near the laptop.
    Mic,
    /// System audio: the other side of a call.
    System,
}

impl Channel {
    pub fn from_name(name: &str) -> Option<Self> {
        match name {
            "mic" => Some(Self::Mic),
            "system" => Some(Self::System),
            _ => None,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Mic => "mic",
            Self::System => "system",
        }
    }

    /// The label its segments carry in `Segment::channel`.
    pub fn label(self) -> &'static str {
        match self {
            Self::Mic => "me",
            Self::System => "them",
        }
    }
}

#[derive(Debug, thiserror::Error)]
pub enum RecorderError {
    #[error("this session is already recording")]
    AlreadyRecording,
    #[error("a session with this id already exists")]
    Exists,
    #[error("this session is not recording")]
    NotRecording,
    #[error("no such session")]
    NotFound,
    #[error("the final pass was cancelled")]
    Cancelled,
    #[error("the final pass failed: {0}")]
    Pass(String),
    #[error("a session needs at least one channel")]
    NoChannels,
    #[error("session ids are letters, digits, '-' and '_'")]
    BadId,
    #[error(transparent)]
    Io(#[from] io::Error),
    #[error(transparent)]
    Store(#[from] StoreError),
}

pub type StartError = RecorderError;
pub type Result<T> = std::result::Result<T, RecorderError>;

struct LiveChannel {
    audio: Arc<PcmStore>,
    pipeline: SessionPipeline,
}

struct Live {
    session: Session,
    channels: Vec<LiveChannel>,
}

pub struct Recorder {
    dir: PathBuf,
    store: Arc<SessionStore>,
    cfg: RecorderConfig,
    live: Mutex<HashMap<String, Live>>,
}

impl Recorder {
    /// Open the data folder. Sessions a previous process left mid-flight are marked
    /// interrupted; their audio and transcript so far stay.
    pub fn open(dir: impl AsRef<Path>, cfg: RecorderConfig) -> Result<Self> {
        let dir = dir.as_ref().to_path_buf();
        let store = SessionStore::open(dir.join("meetbutler.db"))?;
        let recovered = store.recover_interrupted()?;
        if recovered > 0 {
            log::info!("marked {recovered} interrupted session(s)");
        }
        Ok(Self {
            dir,
            store: Arc::new(store),
            cfg,
            live: Mutex::new(HashMap::new()),
        })
    }

    pub fn store(&self) -> &Arc<SessionStore> {
        &self.store
    }

    fn live(&self) -> MutexGuard<'_, HashMap<String, Live>> {
        self.live.lock().unwrap_or_else(|e| e.into_inner())
    }

    fn audio_path(&self, session_id: &str, channel: Channel) -> PathBuf {
        self.dir
            .join("sessions")
            .join(session_id)
            .join(format!("{}.pcm", channel.name()))
    }

    /// Start recording. Returns one PcmStore per channel, in order, for capture to write
    /// into; transcription reads them on its own schedule.
    pub fn start<F, T>(
        &self,
        session_id: &str,
        channels: &[Channel],
        language: Option<String>,
        mut transcriber: F,
    ) -> Result<Vec<Arc<PcmStore>>>
    where
        F: FnMut(Channel) -> T,
        T: Transcribe + Send + 'static,
    {
        if !valid_id(session_id) {
            return Err(RecorderError::BadId);
        }
        if channels.is_empty() {
            return Err(RecorderError::NoChannels);
        }
        let mut live = self.live();
        if live.contains_key(session_id) {
            return Err(RecorderError::AlreadyRecording);
        }
        // reusing an id would append to the old recording and corrupt both
        if self.store.load_session(session_id)?.is_some() {
            return Err(RecorderError::Exists);
        }

        let session = Session {
            sources: channels.iter().map(|c| c.name().to_string()).collect(),
            language,
            ..Session::new(session_id, now())
        };
        self.store.save_session(&session)?;

        let mut running = Vec::with_capacity(channels.len());
        for &channel in channels {
            let audio = Arc::new(PcmStore::open(self.audio_path(session_id, channel))?);
            let t =
                IncrementalTranscriber::new(audio.clone(), transcriber(channel), self.cfg.window)
                    .with_channel(channel.label());
            let pipeline = SessionPipeline::start(t, self.sink(session_id), self.cfg.interval);
            running.push(LiveChannel { audio, pipeline });
        }
        let stores = running.iter().map(|c| c.audio.clone()).collect();
        live.insert(
            session_id.to_string(),
            Live {
                session,
                channels: running,
            },
        );
        Ok(stores)
    }

    /// Persist first: captions are nice to have, a lost transcript is not recoverable.
    fn sink(
        &self,
        session_id: &str,
    ) -> impl FnMut(&[Segment]) -> std::result::Result<(), crate::TranscribeError> + Send + 'static
    {
        let (store, id, notify) = (
            self.store.clone(),
            session_id.to_string(),
            self.cfg.on_segments.clone(),
        );
        move |segments| {
            store.add_segments(&id, segments)?;
            if let Some(notify) = &notify {
                notify(&id, segments);
            }
            Ok(())
        }
    }

    /// Stop capture, transcribe what is left, and close the session.
    pub fn stop(&self, session_id: &str) -> Result<Session> {
        // out of the map first, so the drain below does not hold up other sessions
        let Live {
            mut session,
            channels,
        } = self
            .live()
            .remove(session_id)
            .ok_or(RecorderError::NotRecording)?;
        session.status = SessionStatus::Transcribing;
        self.store.save_session(&session)?;

        let mut duration = 0.0f64;
        for mut c in channels {
            c.audio.close(); // late audio from a capture thread is dropped from here on
            duration = duration.max(c.audio.duration());
            c.pipeline.stop();
        }

        session.status = SessionStatus::Stopped;
        session.stopped_at = Some(now());
        session.duration_sec = duration;
        session.transcript_segments = self.store.load_segments(session_id)?.len() as i64;
        self.store.save_session(&session)?;
        Ok(session)
    }

    pub fn session(&self, session_id: &str) -> Result<Option<Session>> {
        if let Some(live) = self.live().get(session_id) {
            return Ok(Some(live.session.clone()));
        }
        Ok(self.store.load_session(session_id)?)
    }

    pub fn sessions(&self) -> Result<Vec<Session>> {
        Ok(self.store.load_sessions()?)
    }

    /// The transcript in time order across channels: from the running pipelines while a
    /// session is live, from the database once it has stopped. `None` for an unknown id.
    pub fn transcript(&self, session_id: &str) -> Result<Option<Vec<Segment>>> {
        let mut segments = match self.live().get(session_id) {
            Some(live) => live
                .channels
                .iter()
                .flat_map(|c| c.pipeline.transcript())
                .collect(),
            None => {
                if self.store.load_session(session_id)?.is_none() {
                    return Ok(None);
                }
                self.store.load_segments(session_id)?
            }
        };
        segments.sort_by(|a, b| a.start.total_cmp(&b.start));
        Ok(Some(segments))
    }

    /// The after-meeting pass: transcribe every channel again from its recording with
    /// the final model and replace the live transcript with the result in one
    /// transaction. Works on stopped and interrupted sessions alike.
    ///
    /// On failure the live transcript stays and the reason is kept in `error`; when
    /// `cancel` is set (the app is quitting) nothing changes and the pass can run again.
    /// `progress` runs from 0 to 1 across all channels.
    pub fn finalize<V, D>(
        &self,
        session_id: &str,
        mut vad: impl FnMut() -> std::result::Result<V, TranscribeError>,
        mut decoder: impl FnMut(Channel) -> std::result::Result<D, TranscribeError>,
        cancel: &AtomicBool,
        mut progress: impl FnMut(f64),
    ) -> Result<Session>
    where
        V: SpeechProbs,
        D: SpanDecode,
    {
        if self.live().contains_key(session_id) {
            return Err(RecorderError::AlreadyRecording);
        }
        let original = self
            .store
            .load_session(session_id)?
            .ok_or(RecorderError::NotFound)?;
        let mut session = Session {
            status: SessionStatus::Finalizing,
            ..original.clone()
        };
        self.store.save_session(&session)?;

        let channels: Vec<Channel> = session
            .sources
            .iter()
            .filter_map(|s| Channel::from_name(s))
            .collect();
        let n = channels.len().max(1) as f64;
        let mut transcript = Vec::new();
        let mut duration = 0.0f64;
        let mut failure = None;
        for (i, &channel) in channels.iter().enumerate() {
            if cancel.load(Ordering::SeqCst) {
                break;
            }
            let Some(audio) = self.audio(session_id, channel)? else {
                continue;
            };
            duration = duration.max(audio.duration());
            let pass = vad().and_then(|mut vad| {
                let mut decoder = Cancellable {
                    inner: decoder(channel)?,
                    cancel,
                };
                final_transcript(
                    &*audio,
                    &mut vad,
                    &mut decoder,
                    &BENCH,
                    FINAL_CHUNK_SEC,
                    |p| progress((i as f64 + p) / n),
                )
            });
            match pass {
                Ok(segments) => transcript.extend(segments.into_iter().map(|mut s| {
                    s.channel = Some(channel.label().to_string());
                    s
                })),
                Err(err) => {
                    failure = Some(err.to_string());
                    break;
                }
            }
        }

        if cancel.load(Ordering::SeqCst) {
            self.store.save_session(&original)?;
            return Err(RecorderError::Cancelled);
        }
        if let Some(reason) = failure {
            let status = SessionStatus::Stopped;
            self.store.save_session(&Session {
                status,
                error: Some(reason.clone()),
                ..original
            })?;
            return Err(RecorderError::Pass(reason));
        }

        transcript.sort_by(|a, b| a.start.total_cmp(&b.start));
        self.store.replace_transcript(session_id, &transcript)?;
        session.status = SessionStatus::Stopped;
        session.transcript = TranscriptKind::Final;
        session.transcript_segments = transcript.len() as i64;
        session.duration_sec = duration;
        session.stopped_at = session.stopped_at.or(Some(now()));
        session.error = None;
        self.store.save_session(&session)?;
        progress(1.0);
        Ok(session)
    }

    /// A channel's recording, live or finished. `None` if it was never recorded.
    pub fn audio(&self, session_id: &str, channel: Channel) -> Result<Option<Arc<PcmStore>>> {
        if let Some(live) = self.live().get(session_id) {
            let i = live
                .session
                .sources
                .iter()
                .position(|s| s == channel.name());
            return Ok(i.map(|i| live.channels[i].audio.clone()));
        }
        let path = self.audio_path(session_id, channel);
        if !path.exists() {
            return Ok(None);
        }
        let audio = PcmStore::open(path)?;
        audio.close(); // read-only: a finished recording is never appended to
        Ok(Some(Arc::new(audio)))
    }
}

/// Ends a pass at the next span once the app asks to quit.
struct Cancellable<'a, D> {
    inner: D,
    cancel: &'a AtomicBool,
}

impl<D: SpanDecode> SpanDecode for Cancellable<'_, D> {
    fn decode(
        &mut self,
        audio: &[f32],
    ) -> std::result::Result<Vec<crate::incremental::RawSegment>, TranscribeError> {
        if self.cancel.load(Ordering::SeqCst) {
            return Err("cancelled".into());
        }
        self.inner.decode(audio)
    }
}

/// Ids name a folder, so they must not be able to name any other.
fn valid_id(id: &str) -> bool {
    !id.is_empty()
        && id
            .chars()
            .all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_')
}

fn now() -> f64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0.0, |d| d.as_secs_f64())
}
