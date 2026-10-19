//! Per-channel wiring: a worker thread runs transcription passes on a schedule.
//!
//! Capture writes into the PcmStore and returns; it never touches the worker. A slow pass
//! slows captions down but never stops audio from being recorded, and the live transcript
//! can be read at any time without waiting for a pass.

use std::panic::{catch_unwind, AssertUnwindSafe};
use std::sync::{Arc, Condvar, Mutex, MutexGuard};
use std::thread::JoinHandle;
use std::time::{Duration, Instant};

use crate::incremental::{IncrementalTranscriber, Segment};
use crate::{AudioSource, Transcribe, TranscribeError};

/// Bound on how many catch-up passes `stop` will run, so a stuck session can never spin
/// forever at hangup.
const MAX_DRAIN_PASSES: usize = 2000;

/// What the worker drives: one pass at a time.
pub trait Step {
    fn step(&mut self, force: bool) -> Vec<Segment>;
    fn pending_samples(&self) -> usize;
    fn transcript(&self) -> &[Segment];
}

impl<S: AudioSource, T: Transcribe> Step for IncrementalTranscriber<S, T> {
    fn step(&mut self, force: bool) -> Vec<Segment> {
        IncrementalTranscriber::step(self, force)
    }
    fn pending_samples(&self) -> usize {
        IncrementalTranscriber::pending_samples(self)
    }
    fn transcript(&self) -> &[Segment] {
        IncrementalTranscriber::transcript(self)
    }
}

pub type BoxedStep = Box<dyn Step + Send>;

/// Receives each pass's new segments: persist them, then show them.
pub trait Sink: FnMut(&[Segment]) -> Result<(), TranscribeError> + Send {}
impl<F: FnMut(&[Segment]) -> Result<(), TranscribeError> + Send> Sink for F {}

#[derive(Default)]
struct Shared {
    stopped: Mutex<bool>,
    wake: Condvar,
    transcript: Mutex<Vec<Segment>>,
}

fn lock<T>(m: &Mutex<T>) -> MutexGuard<'_, T> {
    m.lock().unwrap_or_else(|e| e.into_inner())
}

/// Owns the transcription worker for one channel of one session.
pub struct SessionPipeline {
    shared: Arc<Shared>,
    worker: Option<JoinHandle<BoxedStep>>,
}

impl SessionPipeline {
    /// Start a worker that runs a pass every `interval` until `stop`.
    pub fn start<T, K>(transcriber: T, sink: K, interval: Duration) -> Self
    where
        T: Step + Send + 'static,
        K: Sink + 'static,
    {
        let shared = Arc::new(Shared::default());
        let worker = {
            let shared = shared.clone();
            std::thread::Builder::new()
                .name("meetbutler-stt".into())
                .spawn(move || run(Box::new(transcriber), sink, interval, &shared))
                .expect("spawn transcription worker")
        };
        Self {
            shared,
            worker: Some(worker),
        }
    }

    /// Segments committed so far. Never waits for a running pass.
    pub fn transcript(&self) -> Vec<Segment> {
        lock(&self.shared.transcript).clone()
    }

    /// Stop the schedule, then transcribe whatever is left with forced passes and hand the
    /// transcriber back. Stop capture first: once this is called no more audio counts.
    /// Returns `None` if already stopped.
    pub fn stop(&mut self) -> Option<BoxedStep> {
        let worker = self.worker.take()?;
        *lock(&self.shared.stopped) = true;
        self.shared.wake.notify_all();
        match worker.join() {
            Ok(t) => Some(t),
            Err(_) => {
                log::error!("transcription worker died");
                None
            }
        }
    }
}

impl Drop for SessionPipeline {
    fn drop(&mut self) {
        self.stop();
    }
}

fn run(mut t: BoxedStep, mut sink: impl Sink, interval: Duration, shared: &Shared) -> BoxedStep {
    loop {
        let deadline = Instant::now() + interval;
        let mut stopped = lock(&shared.stopped);
        while !*stopped {
            let now = Instant::now();
            if now >= deadline {
                break;
            }
            stopped = shared
                .wake
                .wait_timeout(stopped, deadline - now)
                .unwrap_or_else(|e| e.into_inner())
                .0;
        }
        if *stopped {
            break;
        }
        drop(stopped);
        pass(&mut t, &mut sink, false, shared);
    }

    // No more audio is coming, so every drain pass is forced: speech running into the end
    // of the session is committed instead of held back for audio that never arrives.
    let mut passes = 0;
    while t.pending_samples() > 0 {
        if passes == MAX_DRAIN_PASSES {
            log::warn!(
                "gave up draining after {passes} passes, {} samples left",
                t.pending_samples()
            );
            break;
        }
        pass(&mut t, &mut sink, true, shared);
        passes += 1;
    }
    t
}

fn pass(t: &mut BoxedStep, sink: &mut impl Sink, force: bool, shared: &Shared) {
    let segments = match catch_unwind(AssertUnwindSafe(|| t.step(force))) {
        Ok(segments) => segments,
        Err(_) => {
            // a bad pass must not end the session
            log::error!("transcription pass panicked");
            return;
        }
    };
    if segments.is_empty() {
        return;
    }
    lock(&shared.transcript).extend(segments.iter().cloned());
    match catch_unwind(AssertUnwindSafe(|| sink(&segments))) {
        Ok(Ok(())) => {}
        Ok(Err(err)) => log::warn!("delivering {} segments failed: {err}", segments.len()),
        Err(_) => log::error!("delivering {} segments panicked", segments.len()),
    }
}
