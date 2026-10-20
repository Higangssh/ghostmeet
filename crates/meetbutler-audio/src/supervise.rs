//! A capture that is built again when its stream ends under it.
//!
//! A stream on the default device ends when that device changes (a headset plugged in,
//! Bluetooth switching to its call profile) or goes away; cpal says so through the
//! stream's error callback. The callback asks for a rebuild ([`Rebuild::request`]); the
//! capture's thread drops the old stream, builds a new one on whatever the default is
//! now, and tries again every `retry` until one opens or the capture is stopped. Whatever
//! the builds share (the sink, the gap filler) lives outside them, so the channel's
//! clock runs on across the change and the hole is filled with silence.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::mpsc::{self, RecvTimeoutError};
use std::sync::Arc;
use std::time::Duration;

use crate::{Capture, CaptureError};

/// Handed to each build, for its error callback.
#[derive(Clone, Debug, Default)]
pub struct Rebuild(Arc<AtomicBool>);

impl Rebuild {
    /// Ask for the stream to be built again; safe from any thread, cheap, never blocks.
    pub fn request(&self) {
        self.0.store(true, Ordering::SeqCst);
    }

    fn take(&self) -> bool {
        self.0.swap(false, Ordering::SeqCst)
    }
}

/// How often a running stream is asked whether it is stale.
const WATCH_EVERY: Duration = Duration::from_secs(1);

/// Run `build`'s result on a thread of its own until stopped, building it again when asked.
/// The first build's error is the open's error; later ones are logged and retried.
pub fn supervised<T: 'static>(
    name: &str,
    retry: Duration,
    build: impl FnMut(&Rebuild) -> Result<T, CaptureError> + Send + 'static,
) -> Result<Box<dyn Capture>, CaptureError> {
    supervised_watching(name, retry, build, |_: &T| false)
}

/// As [`supervised`], and also built again when `stale` says so, asked about once a
/// second: for a change no callback reports (on macOS an input stream stays on its
/// device when the default input changes).
pub fn supervised_watching<T: 'static>(
    name: &str,
    retry: Duration,
    mut build: impl FnMut(&Rebuild) -> Result<T, CaptureError> + Send + 'static,
    mut stale: impl FnMut(&T) -> bool + Send + 'static,
) -> Result<Box<dyn Capture>, CaptureError> {
    let (ready_tx, ready_rx) = mpsc::channel();
    let (stop_tx, stop_rx) = mpsc::channel::<()>();
    let label = name.to_string();
    let handle = std::thread::Builder::new()
        .name(format!("meetbutler-{name}"))
        .spawn(move || {
            let rebuild = Rebuild::default();
            let mut current = match build(&rebuild) {
                Ok(t) => {
                    let _ = ready_tx.send(Ok(()));
                    Some(t)
                }
                Err(err) => {
                    let _ = ready_tx.send(Err(err));
                    return;
                }
            };
            // poll often enough that a device change costs little audio
            let tick = retry.min(Duration::from_millis(100));
            let (mut since_try, mut since_watch) = (Duration::ZERO, Duration::ZERO);
            loop {
                match stop_rx.recv_timeout(tick) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
                since_watch += tick;
                if since_watch >= WATCH_EVERY {
                    since_watch = Duration::ZERO;
                    if current.as_ref().is_some_and(&mut stale) {
                        rebuild.request();
                    }
                }
                if rebuild.take() {
                    log::info!(
                        "{label}: the stream ended or its device changed; building it again"
                    );
                    current = None; // the old stream goes before the new one opens
                    since_try = retry; // try at once
                }
                if current.is_none() {
                    since_try += tick;
                    if since_try >= retry {
                        since_try = Duration::ZERO;
                        match build(&rebuild) {
                            Ok(t) => current = Some(t),
                            Err(err) => log::warn!("{label}: not back yet: {err}"),
                        }
                    }
                }
            }
            drop(current);
        })
        .map_err(|e| CaptureError::Backend(e.to_string()))?;
    ready_rx
        .recv()
        .map_err(|_| CaptureError::Backend(format!("{name} thread died")))??;
    Ok(Box::new(crate::ThreadCapture {
        stop: Some(stop_tx),
        handle: Some(handle),
    }))
}
