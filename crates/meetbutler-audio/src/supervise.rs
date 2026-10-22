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
use std::time::{Duration, Instant};

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
pub const WATCH_EVERY: Duration = Duration::from_secs(1);

/// Run `build`'s result on a thread of its own until stopped, building it again when asked.
/// The first build's error is the open's error; later ones are logged and retried.
pub fn supervised<T: 'static>(
    name: &str,
    retry: Duration,
    build: impl FnMut(&Rebuild) -> Result<T, CaptureError> + Send + 'static,
) -> Result<Box<dyn Capture>, CaptureError> {
    supervised_watching(name, retry, WATCH_EVERY, build, |_: &T| false)
}

/// As [`supervised`], and also built again when `stale` says so, asked every
/// `watch_every` (the app uses [`WATCH_EVERY`]): for a change no callback reports (on
/// macOS an input stream stays on its device when the default input changes).
pub fn supervised_watching<T: 'static>(
    name: &str,
    retry: Duration,
    watch_every: Duration,
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
            let mut timers = Timers::new(Instant::now(), retry, watch_every);
            loop {
                match stop_rx.recv_timeout(tick) {
                    Ok(()) | Err(RecvTimeoutError::Disconnected) => break,
                    Err(RecvTimeoutError::Timeout) => {}
                }
                if timers.watch_due(Instant::now()) && current.as_ref().is_some_and(&mut stale) {
                    rebuild.request();
                }
                if rebuild.take() {
                    log::info!(
                        "{label}: the stream ended or its device changed; building it again"
                    );
                    current = None; // the old stream goes before the new one opens
                    timers.try_at_once();
                }
                if current.is_none() && timers.retry_due(Instant::now()) {
                    match build(&rebuild) {
                        Ok(t) => current = Some(t),
                        Err(err) => log::warn!("{label}: not back yet: {err}"),
                    }
                    timers.tried(Instant::now());
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

/// When the supervising loop next asks `stale` and next tries a build. By the clock, not
/// by counting ticks: on a busy machine a tick takes longer than asked.
struct Timers {
    retry: Duration,
    watch_every: Duration,
    watched: Instant,
    tried: Option<Instant>, // None: try at the next tick
}

impl Timers {
    fn new(now: Instant, retry: Duration, watch_every: Duration) -> Self {
        Timers {
            retry,
            watch_every,
            watched: now,
            tried: Some(now),
        }
    }

    fn watch_due(&mut self, now: Instant) -> bool {
        if now.saturating_duration_since(self.watched) < self.watch_every {
            return false;
        }
        self.watched = now;
        true
    }

    fn try_at_once(&mut self) {
        self.tried = None;
    }

    /// Since the last try ended: a slow build does not eat into the wait after it.
    fn retry_due(&self, now: Instant) -> bool {
        self.tried
            .is_none_or(|at| now.saturating_duration_since(at) >= self.retry)
    }

    fn tried(&mut self, now: Instant) {
        self.tried = Some(now);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    const MS: fn(u64) -> Duration = Duration::from_millis;

    #[test]
    fn a_retry_is_due_by_the_clock_however_long_each_tick_took() {
        let t0 = Instant::now();
        let mut t = Timers::new(t0, MS(1000), MS(1000));
        t.tried(t0); // a build failed
        assert!(!t.retry_due(t0 + MS(600)), "a slow tick, not a second yet");
        assert!(
            t.retry_due(t0 + MS(1200)),
            "two slow ticks, past the second"
        );
        t.tried(t0 + MS(1200));
        assert!(!t.retry_due(t0 + MS(1300)), "not again right after a try");
    }

    #[test]
    fn a_requested_rebuild_is_tried_at_once() {
        let t0 = Instant::now();
        let mut t = Timers::new(t0, MS(60_000), MS(1000));
        t.tried(t0);
        t.try_at_once();
        assert!(t.retry_due(t0));
    }

    #[test]
    fn the_watch_is_due_every_period_by_the_clock() {
        let t0 = Instant::now();
        let mut t = Timers::new(t0, MS(20), MS(1000));
        assert!(!t.watch_due(t0 + MS(999)));
        assert!(t.watch_due(t0 + MS(1000)));
        assert!(
            !t.watch_due(t0 + MS(1500)),
            "counted again from the last one"
        );
        assert!(
            t.watch_due(t0 + MS(3000)),
            "one slow tick is still one check"
        );
    }
}
