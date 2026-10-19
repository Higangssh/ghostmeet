//! Microphone and system-audio capture for MeetButler.
//!
//! Each source is delivered on its own, as 16 kHz mono f32, to a sink the caller supplies;
//! the engine records them as separate channels ("me" and "them"). Nothing here starts on
//! its own: capture begins only when the app asks, after the user pressed record.
//!
//! Platforms: the microphone through cpal everywhere; system audio through
//! ScreenCaptureKit on macOS 13+. Windows system audio (WASAPI loopback) has its place in
//! [`system`] but is not wired up yet, and reports [`CaptureError::Unsupported`].

pub mod conform;
pub mod mic;
pub mod system;

/// The rate every sink receives.
pub const TARGET_RATE: u32 = 16_000;

#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum Source {
    Mic,
    System,
}

/// Receives 16 kHz mono f32 from a capture thread. Must not block for long.
pub type AudioSink = Box<dyn FnMut(&[f32]) + Send>;

#[derive(Debug, thiserror::Error)]
pub enum CaptureError {
    #[error("permission needed: {0}")]
    PermissionDenied(String),
    #[error("no input device")]
    NoDevice,
    #[error("not supported on this platform yet: {0}")]
    Unsupported(&'static str),
    #[error("{0}")]
    Backend(String),
}

/// A running capture. Stopping is idempotent; dropping stops it.
pub trait Capture: Send {
    fn stop(&mut self);
}

/// Open one source with the platform backend.
pub fn open(source: Source, sink: AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    match source {
        Source::Mic => mic::open(sink),
        Source::System => system::open(sink),
    }
}

/// The sources that opened, and why the others did not.
pub struct Opened {
    pub captures: Vec<(Source, Box<dyn Capture>)>,
    pub failed: Vec<(Source, CaptureError)>,
}

impl Opened {
    pub fn sources(&self) -> Vec<Source> {
        self.captures.iter().map(|(s, _)| *s).collect()
    }

    pub fn stop(&mut self) {
        for (_, c) in &mut self.captures {
            c.stop();
        }
        self.captures.clear();
    }
}

impl Drop for Opened {
    fn drop(&mut self) {
        self.stop();
    }
}

/// Open every requested source that can be opened; a meeting records with what it has.
/// Fails only when none opened, with every reason.
pub fn open_sources(
    requested: &[Source],
    mut sink: impl FnMut(Source) -> AudioSink,
    mut backend: impl FnMut(Source, AudioSink) -> Result<Box<dyn Capture>, CaptureError>,
) -> Result<Opened, Vec<(Source, CaptureError)>> {
    let mut opened = Opened {
        captures: Vec::new(),
        failed: Vec::new(),
    };
    let mut seen = Vec::new();
    for &source in requested {
        if seen.contains(&source) {
            continue;
        }
        seen.push(source);
        match backend(source, sink(source)) {
            Ok(capture) => opened.captures.push((source, capture)),
            Err(err) => {
                log::warn!("{source:?} not recording: {err}");
                opened.failed.push((source, err));
            }
        }
    }
    if opened.captures.is_empty() {
        return Err(std::mem::take(&mut opened.failed));
    }
    Ok(opened)
}

/// Keep a backend's stream object on a thread of its own until stopped. Some backends'
/// streams may not leave the thread that built them; this way none has to.
pub(crate) fn on_own_thread<T: 'static>(
    name: &str,
    build: impl FnOnce() -> Result<T, CaptureError> + Send + 'static,
) -> Result<Box<dyn Capture>, CaptureError> {
    use std::sync::mpsc;

    let (ready_tx, ready_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let handle = std::thread::Builder::new()
        .name(format!("meetbutler-{name}"))
        .spawn(move || match build() {
            Ok(stream) => {
                let _ = ready_tx.send(Ok(()));
                let _ = stop_rx.recv(); // a stop, or the handle going away
                drop(stream);
            }
            Err(err) => {
                let _ = ready_tx.send(Err(err));
            }
        })
        .map_err(|e| CaptureError::Backend(e.to_string()))?;
    ready_rx
        .recv()
        .map_err(|_| CaptureError::Backend(format!("{name} thread died")))??;
    Ok(Box::new(ThreadCapture {
        stop: Some(stop_tx),
        handle: Some(handle),
    }))
}

struct ThreadCapture {
    stop: Option<std::sync::mpsc::Sender<()>>,
    handle: Option<std::thread::JoinHandle<()>>,
}

impl Capture for ThreadCapture {
    fn stop(&mut self) {
        if let Some(stop) = self.stop.take() {
            let _ = stop.send(());
        }
        if let Some(handle) = self.handle.take() {
            let _ = handle.join();
        }
    }
}

impl Drop for ThreadCapture {
    fn drop(&mut self) {
        self.stop();
    }
}
