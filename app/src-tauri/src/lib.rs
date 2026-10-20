//! MeetButler desktop app: a thin shell over `meetbutler-engine` and `meetbutler-audio`.
//!
//! The app opens no port. The window talks to Rust over Tauri's IPC only, and the only
//! audio that leaves a capture callback goes into a PcmStore in the data folder.
//!
//! After Stop: the final pass, then the summary, one after the other on one background
//! thread. The summary runs in the sidecar (`meetbutler-summarizer`, shipped next to the
//! app's executable), which is started for it and exits when it is done.

pub mod markdown;

use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc;
use std::sync::{Arc, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{SystemTime, UNIX_EPOCH};

use meetbutler_audio::{open_sources, Opened, Source};
use meetbutler_engine::incremental::Segment;
use meetbutler_engine::model_cache::ModelCache;
use meetbutler_engine::recorder::{Channel, Recorder, RecorderConfig};
use meetbutler_engine::silero::SileroOnnx;
use meetbutler_engine::span_decoder::SpanTranscriber;
use meetbutler_engine::spans::BENCH;
use meetbutler_engine::store::{Session, Summary as StoredSummary};
use meetbutler_engine::whisper::{WhisperDecoder, WhisperModel};
use meetbutler_summarizer::client::{ClientError, Sidecar};
use meetbutler_summarizer::protocol::ErrorKind;
use meetbutler_summarizer::summary::{self, Summary};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

/// Provisional captions during the meeting; the final transcript after it.
pub const LIVE_MODEL: &str = "small";
pub const FINAL_MODEL: &str = "large-v3-turbo";
pub const VAD_MODEL: &str = "silero_vad.onnx";
/// The local summary model (`bench/results/RESULTS.md` §8), in the models folder.
pub const SUMMARY_MODEL: &str = "qwen3.5-4b";
pub const SUMMARY_MODEL_FILE: &str = "qwen3.5-4b-Q4_K_M.gguf";
const SIDECAR: &str = "meetbutler-summarizer";

/// The sidecar: `MEETBUTLER_SUMMARIZER`, else next to the app's own executable (where the
/// bundle puts it, and where cargo builds it).
fn sidecar_path() -> Result<PathBuf, String> {
    if let Some(p) = std::env::var_os("MEETBUTLER_SUMMARIZER") {
        return Ok(PathBuf::from(p));
    }
    let exe = std::env::current_exe().map_err(|e| e.to_string())?;
    Ok(exe.with_file_name(SIDECAR))
}

/// Where the ggml models are: `MEETBUTLER_MODELS`, else `<data folder>/models`. They are
/// downloaded ahead of time; the app does not fetch them.
fn models_dir(data: &std::path::Path) -> PathBuf {
    std::env::var_os("MEETBUTLER_MODELS").map_or_else(|| data.join("models"), PathBuf::from)
}

pub struct AppState {
    recorder: Arc<Recorder>,
    models_dir: PathBuf,
    models: Arc<ModelCache<String, WhisperModel>>,
    current: Mutex<Option<Current>>,
    /// The after-meeting work (final pass, then summary), at most one at a time.
    finalizing: Mutex<Option<JoinHandle<()>>>,
    quitting: Arc<AtomicBool>,
    /// Set to stop the summary that is running; cleared when the next one starts.
    cancel_summary: Arc<AtomicBool>,
}

struct Current {
    session_id: String,
    capture: Opened,
    writers: Vec<JoinHandle<()>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

impl AppState {
    pub fn open(
        data: PathBuf,
        on_segments: impl Fn(&str, &[Segment]) + Send + Sync + 'static,
    ) -> Result<Self, String> {
        let cfg = RecorderConfig {
            on_segments: Some(Arc::new(on_segments)),
            ..RecorderConfig::default()
        };
        let recorder = Recorder::open(&data, cfg).map_err(|e| e.to_string())?;
        Ok(Self {
            recorder: Arc::new(recorder),
            models_dir: models_dir(&data),
            models: Arc::new(ModelCache::new()),
            current: Mutex::new(None),
            finalizing: Mutex::new(None),
            quitting: Arc::new(AtomicBool::new(false)),
            cancel_summary: Arc::new(AtomicBool::new(false)),
        })
    }

    fn model(&self, name: &str) -> Result<Arc<WhisperModel>, String> {
        load_model(&self.models, &self.models_dir, name)
    }

    /// Stop capture, drain the live pass and close the session. No final pass.
    fn stop_current(&self) -> Result<Option<Session>, String> {
        let Some(mut current) = lock(&self.current).take() else {
            return Ok(None);
        };
        current.capture.stop(); // capture threads end, their sinks drop, writers see the end
        for w in current.writers {
            let _ = w.join();
        }
        self.recorder
            .stop(&current.session_id)
            .map(Some)
            .map_err(|e| e.to_string())
    }

    /// Before the process exits: stop recording, cancel the final pass, and let go of
    /// every model. A Metal model still alive at exit aborts the process (see
    /// `WhisperModel`), and an abort could cost the end of the recording.
    pub fn shutdown(&self) {
        self.quitting.store(true, Ordering::SeqCst);
        self.cancel_summary.store(true, Ordering::SeqCst);
        if let Err(err) = self.stop_current() {
            log::error!("stopping the recording at exit: {err}");
        }
        if let Some(pass) = lock(&self.finalizing).take() {
            let _ = pass.join();
        }
        self.models.clear();
    }
}

fn load_model(
    cache: &ModelCache<String, WhisperModel>,
    dir: &std::path::Path,
    name: &str,
) -> Result<Arc<WhisperModel>, String> {
    cache.get_or_load(name.to_string(), |n| {
        let path = dir.join(format!("ggml-{n}.bin"));
        if !path.exists() {
            return Err(format!("model not found: {}", path.display()));
        }
        WhisperModel::load(&path, true).map_err(|e| e.to_string())
    })
}

fn source_channel(s: Source) -> Channel {
    match s {
        Source::Mic => Channel::Mic,
        Source::System => Channel::System,
    }
}

fn now_ms() -> u128 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map_or(0, |d| d.as_millis())
}

#[derive(Serialize)]
struct Started {
    session_id: String,
    sources: Vec<String>,
    /// Sources asked for that could not be opened, and why.
    warnings: Vec<String>,
}

#[derive(Clone, Serialize)]
struct SegmentsEvent {
    session_id: String,
    segments: Vec<Segment>,
}

#[derive(Clone, Serialize)]
struct SummaryEvent {
    session_id: String,
    progress: f64,
    done: bool,
    cancelled: bool,
    error: Option<String>,
}

#[derive(Clone, Serialize)]
struct FinalEvent {
    session_id: String,
    progress: f64,
    done: bool,
    error: Option<String>,
}

/// Start recording from `sources` ("mic", "system"). Records with whatever opens; the
/// rest come back as warnings. Off the main thread: loading the model takes a moment.
#[tauri::command(async)]
fn start_recording(
    app: AppHandle,
    state: State<'_, AppState>,
    sources: Vec<String>,
    language: Option<String>,
    vocabulary: Option<Vec<String>>,
) -> Result<Started, String> {
    let mut current = lock(&state.current);
    if current.is_some() {
        return Err("already recording".into());
    }
    let requested: Vec<Source> = sources
        .iter()
        .map(|s| match s.as_str() {
            "mic" => Ok(Source::Mic),
            "system" => Ok(Source::System),
            other => Err(format!("unknown source {other}")),
        })
        .collect::<Result<_, _>>()?;
    let language = language.filter(|l| !l.is_empty() && l != "auto");
    let vocabulary: Vec<String> = vocabulary
        .unwrap_or_default()
        .into_iter()
        .map(|t| t.trim().to_string())
        .filter(|t| !t.is_empty())
        .collect();

    let model = state.model(LIVE_MODEL)?;
    let vad_path = state.models_dir.join(VAD_MODEL);

    // capture first, into channels: the session is created with what actually opened
    let mut feeds = Vec::new();
    let capture = open_sources(
        &requested,
        |source| {
            let (tx, rx) = mpsc::channel::<Vec<f32>>();
            feeds.push((source, rx));
            Box::new(move |audio: &[f32]| {
                let _ = tx.send(audio.to_vec());
            })
        },
        meetbutler_audio::open,
    )
    .map_err(|failed| {
        let reasons: Vec<String> = failed.iter().map(|(s, e)| format!("{s:?}: {e}")).collect();
        format!("nothing to record from ({})", reasons.join("; "))
    })?;
    let opened = capture.sources();
    let warnings = capture
        .failed
        .iter()
        .map(|(s, e)| format!("{s:?}: {e}"))
        .collect();

    let channels: Vec<Channel> = opened.iter().map(|&s| source_channel(s)).collect();
    let mut transcribers = Vec::new();
    for _ in &channels {
        let vad = SileroOnnx::load(&vad_path).map_err(|e| e.to_string())?;
        let decoder = WhisperDecoder::new(model.clone(), language.clone())
            .map_err(|e| e.to_string())?
            .with_vocabulary(&vocabulary);
        transcribers.push(Some(SpanTranscriber::new(vad, decoder, BENCH)));
    }
    let session_id = format!("m{}", now_ms());
    let stores = state
        .recorder
        .start(&session_id, &channels, language, |c| {
            let i = channels
                .iter()
                .position(|&x| x == c)
                .expect("a channel we asked for");
            transcribers[i].take().expect("one transcriber per channel")
        })
        .map_err(|e| e.to_string())?;
    if let Err(err) = state.recorder.set_vocabulary(&session_id, &vocabulary) {
        log::error!("{session_id}: keeping the vocabulary failed: {err}");
    }

    let writers = feeds
        .into_iter()
        .filter(|(s, _)| opened.contains(s))
        .map(|(source, rx)| {
            let i = opened.iter().position(|&s| s == source).unwrap();
            let store = stores[i].clone();
            std::thread::spawn(move || {
                for chunk in rx {
                    if let Err(err) = store.append_f32(&chunk) {
                        log::error!("{source:?}: writing audio failed: {err}");
                    }
                }
            })
        })
        .collect();

    *current = Some(Current {
        session_id: session_id.clone(),
        capture,
        writers,
    });
    let _ = app.emit("recording", Some(session_id.clone()));
    Ok(Started {
        session_id,
        sources: channels.iter().map(|c| c.name().to_string()).collect(),
        warnings,
    })
}

/// Stop recording, then run the final pass in the background ("final" events). Off the
/// main thread: draining the last live pass takes a moment.
#[tauri::command(async)]
fn stop_recording(app: AppHandle, state: State<'_, AppState>) -> Result<Option<Session>, String> {
    let session = state.stop_current()?;
    let _ = app.emit("recording", None::<String>);
    if let Some(s) = &session {
        // the live model is not needed until the next meeting; make room for the large one
        state.models.evict(&LIVE_MODEL.to_string());
        start_final_pass(&app, &state, s.session_id.clone());
    }
    Ok(session)
}

/// Run `work` on the after-meeting thread, after whatever is running there now.
fn after_meeting(state: &AppState, work: impl FnOnce() + Send + 'static) {
    let mut slot = lock(&state.finalizing);
    if let Some(previous) = slot.take() {
        let _ = previous.join(); // one at a time: the models are big
    }
    *slot = Some(std::thread::spawn(work));
}

fn start_final_pass(app: &AppHandle, state: &AppState, session_id: String) {
    let (app, recorder, models, dir, quitting) = (
        app.clone(),
        state.recorder.clone(),
        state.models.clone(),
        state.models_dir.clone(),
        state.quitting.clone(),
    );
    let cancel = state.cancel_summary.clone();
    after_meeting(state, move || {
        let emit = |progress: f64, done: bool, error: Option<String>| {
            let _ = app.emit(
                "final",
                FinalEvent {
                    session_id: session_id.clone(),
                    progress,
                    done,
                    error,
                },
            );
        };
        let result = load_model(&models, &dir, FINAL_MODEL).and_then(|model| {
            let session = recorder.session(&session_id).ok().flatten();
            let language = session.as_ref().and_then(|s| s.language.clone());
            let vocabulary = session.map(|s| s.vocabulary).unwrap_or_default();
            recorder
                .finalize(
                    &session_id,
                    || SileroOnnx::load(dir.join(VAD_MODEL)),
                    |_| {
                        WhisperDecoder::new(model.clone(), language.clone())
                            .map(|d| d.with_vocabulary(&vocabulary))
                    },
                    &quitting,
                    |p| emit(p, false, None),
                )
                .map_err(|e| e.to_string())
        });
        // keep the large model only for the pass: the next meeting needs the memory
        models.evict(&FINAL_MODEL.to_string());
        match result {
            Ok(_) => {
                emit(1.0, true, None);
                run_summary(&app, &recorder, &dir, &session_id, &cancel, &quitting);
            }
            Err(err) => emit(0.0, true, Some(err)),
        }
    });
}

/// Summarise a session's transcript through the sidecar and keep the result. Emits
/// "summary" events; a failure or a cancel is kept with the session, not raised.
fn run_summary(
    app: &AppHandle,
    recorder: &Recorder,
    models_dir: &std::path::Path,
    session_id: &str,
    cancel: &AtomicBool,
    quitting: &AtomicBool,
) {
    // a fresh summary is not cancelled; shutdown sets quitting before cancel, so a quit
    // that lands around this line is still seen
    cancel.store(false, Ordering::SeqCst);
    if quitting.load(Ordering::SeqCst) {
        return;
    }
    let emit = |progress: f64, done: bool, cancelled: bool, error: Option<String>| {
        let _ = app.emit(
            "summary",
            SummaryEvent {
                session_id: session_id.to_string(),
                progress,
                done,
                cancelled,
                error,
            },
        );
    };
    let result = (|| -> Result<Summary, ClientError> {
        let lines: Vec<String> = recorder
            .transcript(session_id)
            .map_err(|e| ClientError::Io(std::io::Error::other(e.to_string())))?
            .unwrap_or_default()
            .into_iter()
            .map(|s| s.text)
            .collect();
        let model = models_dir.join(SUMMARY_MODEL_FILE);
        if !model.exists() {
            return Err(ClientError::Failed {
                kind: ErrorKind::Model,
                message: format!("summary model not found: {}", model.display()),
            });
        }
        let program = sidecar_path().map_err(|e| ClientError::Io(std::io::Error::other(e)))?;
        let mut sidecar = Sidecar::spawn(&program, &[])?;
        summary::summarise(
            &mut sidecar,
            &lines,
            &summary::Options { model },
            cancel,
            |p| emit(p, false, false, None),
        )
        // dropping the sidecar here ends the process and gives its memory back
    })();
    let mut stored = StoredSummary::new(session_id);
    stored.model = SUMMARY_MODEL.into();
    let event = match result {
        Ok(s) => {
            stored.input_tokens = s.input_tokens as i64;
            stored.output_tokens = s.output_tokens as i64;
            stored.content = serde_json::to_string(&s).unwrap_or_default();
            (1.0, true, false, None)
        }
        Err(ClientError::Failed {
            kind: ErrorKind::Cancelled,
            ..
        }) => {
            stored.status = "cancelled".into();
            (0.0, true, true, None)
        }
        Err(err) => {
            stored.status = "error".into();
            stored.error = Some(err.to_string());
            (0.0, true, false, Some(err.to_string()))
        }
    };
    if let Err(err) = recorder.store().save_summary(&stored) {
        log::error!("{session_id}: saving the summary failed: {err}");
    }
    emit(event.0, event.1, event.2, event.3);
}

/// Summarise a session again (after a cancel, a failure, or a newer model).
#[tauri::command]
fn summarize(app: AppHandle, state: State<'_, AppState>, session_id: String) {
    let (recorder, dir, cancel, quitting) = (
        state.recorder.clone(),
        state.models_dir.clone(),
        state.cancel_summary.clone(),
        state.quitting.clone(),
    );
    after_meeting(&state, move || {
        run_summary(&app, &recorder, &dir, &session_id, &cancel, &quitting)
    });
}

/// Stop the summary that is running; what it had is not kept.
#[tauri::command]
fn cancel_summary(state: State<'_, AppState>) {
    state.cancel_summary.store(true, Ordering::SeqCst);
}

/// The kept summary of a session, if any: its status and, when done, the fixed format.
#[tauri::command]
fn get_summary(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<Option<StoredSummary>, String> {
    state
        .recorder
        .store()
        .load_summary(&session_id)
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn list_sessions(state: State<'_, AppState>) -> Result<Vec<Session>, String> {
    state.recorder.sessions().map_err(|e| e.to_string())
}

#[tauri::command]
fn get_transcript(
    state: State<'_, AppState>,
    session_id: String,
) -> Result<Option<Vec<Segment>>, String> {
    state
        .recorder
        .transcript(&session_id)
        .map_err(|e| e.to_string())
}

/// Bring one of the app's own windows forward ("panel", "library").
#[tauri::command]
fn show_window(app: AppHandle, label: String) -> Result<(), String> {
    if !matches!(label.as_str(), "panel" | "library") {
        return Err(format!("no window {label}"));
    }
    let w = app
        .get_webview_window(&label)
        .ok_or_else(|| format!("no window {label}"))?;
    w.show()
        .and_then(|_| w.unminimize())
        .and_then(|_| w.set_focus())
        .map_err(|e| e.to_string())
}

#[tauri::command]
fn rename_session(
    app: AppHandle,
    state: State<'_, AppState>,
    session_id: String,
    title: String,
) -> Result<(), String> {
    state
        .recorder
        .rename(&session_id, &title)
        .map_err(|e| e.to_string())?;
    let _ = app.emit("sessions", ());
    Ok(())
}

/// Delete a meeting for good, its recording included. Not while it is recording or in
/// the after-meeting pass.
#[tauri::command]
fn delete_session(
    app: AppHandle,
    state: State<'_, AppState>,
    session_id: String,
) -> Result<(), String> {
    state
        .recorder
        .delete(&session_id)
        .map_err(|e| e.to_string())?;
    let _ = app.emit("sessions", ());
    Ok(())
}

fn meeting_markdown(
    state: &AppState,
    session_id: &str,
    date: &str,
) -> Result<(Session, String), String> {
    let session = state
        .recorder
        .session(session_id)
        .map_err(|e| e.to_string())?
        .ok_or("no such meeting")?;
    let transcript = state
        .recorder
        .transcript(session_id)
        .map_err(|e| e.to_string())?
        .unwrap_or_default();
    let summary: Option<Summary> = state
        .recorder
        .store()
        .load_summary(session_id)
        .map_err(|e| e.to_string())?
        .filter(|s| s.status == "done")
        .and_then(|s| serde_json::from_str(&s.content).ok());
    let md = markdown::markdown(&session, date, &transcript, summary.as_ref());
    Ok((session, md))
}

/// The meeting as Markdown (OUT-1), for the clipboard. `date` is its start as the window
/// shows it.
#[tauri::command]
fn get_markdown(
    state: State<'_, AppState>,
    session_id: String,
    date: String,
) -> Result<String, String> {
    meeting_markdown(&state, &session_id, &date).map(|m| m.1)
}

/// Write the meeting as Markdown into Downloads and show it in Finder. Returns the path.
#[tauri::command]
fn export_markdown(
    app: AppHandle,
    state: State<'_, AppState>,
    session_id: String,
    date: String,
) -> Result<String, String> {
    let (session, md) = meeting_markdown(&state, &session_id, &date)?;
    let dir = app.path().download_dir().map_err(|e| e.to_string())?;
    let name: String = match session.title.trim() {
        "" => format!("MeetButler {}", session.session_id),
        t => t.to_string(),
    }
    .chars()
    .map(|c| {
        if "/\\:*?\"<>|".contains(c) || c.is_control() {
            '_'
        } else {
            c
        }
    })
    .collect();
    let mut path = dir.join(format!("{name}.md"));
    let mut n = 1;
    while path.exists() {
        n += 1;
        path = dir.join(format!("{name} ({n}).md"));
    }
    std::fs::write(&path, md).map_err(|e| e.to_string())?;
    // Finder, not a URL: the file is on this Mac
    let _ = std::process::Command::new("open")
        .arg("-R")
        .arg(&path)
        .status();
    Ok(path.display().to_string())
}

/// The vocabulary of the latest meeting that had one, to start the next one with.
#[tauri::command]
fn last_vocabulary(state: State<'_, AppState>) -> Result<Vec<String>, String> {
    let sessions = state.recorder.sessions().map_err(|e| e.to_string())?;
    Ok(sessions
        .into_iter()
        .rev()
        .map(|s| s.vocabulary)
        .find(|v| !v.is_empty())
        .unwrap_or_default())
}

#[tauri::command]
fn recording(state: State<'_, AppState>) -> Option<String> {
    lock(&state.current).as_ref().map(|c| c.session_id.clone())
}

pub fn run() {
    let app = tauri::Builder::default()
        .setup(|app| {
            // MEETBUTLER_DATA_DIR keeps a development or screenshot run away from real meetings
            let data = match std::env::var_os("MEETBUTLER_DATA_DIR") {
                Some(dir) => PathBuf::from(dir),
                None => app.path().app_data_dir()?,
            };
            let handle = app.handle().clone();
            let state = AppState::open(data, move |id, segs| {
                let _ = handle.emit(
                    "segments",
                    SegmentsEvent {
                        session_id: id.into(),
                        segments: segs.to_vec(),
                    },
                );
            })?;
            app.manage(state);
            Ok(())
        })
        .invoke_handler(tauri::generate_handler![
            start_recording,
            stop_recording,
            list_sessions,
            get_transcript,
            recording,
            summarize,
            cancel_summary,
            get_summary,
            last_vocabulary,
            show_window,
            rename_session,
            delete_session,
            get_markdown,
            export_markdown
        ])
        .on_window_event(|window, event| {
            if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                if window.label() == "main" {
                    // the mini bar is the app: closing it quits, through RunEvent::Exit
                    window.app_handle().exit(0);
                } else {
                    // the panel and the library come back from the mini bar
                    api.prevent_close();
                    let _ = window.hide();
                }
            }
        })
        .build(tauri::generate_context!())
        .expect("building the app");

    app.run(|handle, event| {
        // every way out: Cmd+Q, closing the last window, logging out
        if let RunEvent::Exit = event {
            if let Some(state) = handle.try_state::<AppState>() {
                state.shutdown();
            }
        }
    });
}
