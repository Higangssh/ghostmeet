//! MeetButler desktop app: a thin shell over `meetbutler-engine` and `meetbutler-audio`.
//!
//! The app opens no port. The window talks to Rust over Tauri's IPC only, and the only
//! audio that leaves a capture callback goes into a PcmStore in the data folder.

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
use meetbutler_engine::store::Session;
use meetbutler_engine::whisper::{WhisperDecoder, WhisperModel};
use serde::Serialize;
use tauri::{AppHandle, Emitter, Manager, RunEvent, State};

/// Provisional captions during the meeting; the final transcript after it.
pub const LIVE_MODEL: &str = "small";
pub const FINAL_MODEL: &str = "large-v3-turbo";
pub const VAD_MODEL: &str = "silero_vad.onnx";

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
    /// The after-meeting pass, at most one at a time.
    finalizing: Mutex<Option<JoinHandle<()>>>,
    quitting: Arc<AtomicBool>,
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
    state: State<'_, AppState>,
    sources: Vec<String>,
    language: Option<String>,
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
        let decoder =
            WhisperDecoder::new(model.clone(), language.clone()).map_err(|e| e.to_string())?;
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
    if let Some(s) = &session {
        // the live model is not needed until the next meeting; make room for the large one
        state.models.evict(&LIVE_MODEL.to_string());
        start_final_pass(&app, &state, s.session_id.clone());
    }
    Ok(session)
}

fn start_final_pass(app: &AppHandle, state: &AppState, session_id: String) {
    let mut slot = lock(&state.finalizing);
    if let Some(previous) = slot.take() {
        let _ = previous.join(); // one at a time: the large model is big
    }
    let (app, recorder, models, dir, quitting) = (
        app.clone(),
        state.recorder.clone(),
        state.models.clone(),
        state.models_dir.clone(),
        state.quitting.clone(),
    );
    *slot = Some(std::thread::spawn(move || {
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
            let language = recorder
                .session(&session_id)
                .ok()
                .flatten()
                .and_then(|s| s.language);
            recorder
                .finalize(
                    &session_id,
                    || SileroOnnx::load(dir.join(VAD_MODEL)),
                    |_| WhisperDecoder::new(model.clone(), language.clone()),
                    &quitting,
                    |p| emit(p, false, None),
                )
                .map_err(|e| e.to_string())
        });
        // keep the large model only for the pass: the next meeting needs the memory
        models.evict(&FINAL_MODEL.to_string());
        match result {
            Ok(_) => emit(1.0, true, None),
            Err(err) => emit(0.0, true, Some(err)),
        }
    }));
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

#[tauri::command]
fn recording(state: State<'_, AppState>) -> Option<String> {
    lock(&state.current).as_ref().map(|c| c.session_id.clone())
}

pub fn run() {
    let app = tauri::Builder::default()
        .setup(|app| {
            let data = app.path().app_data_dir()?;
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
            recording
        ])
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
