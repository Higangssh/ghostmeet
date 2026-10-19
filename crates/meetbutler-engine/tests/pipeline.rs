//! SessionPipeline — transcription on its own thread and schedule.
//!
//! Capture writes into the PcmStore and never waits on Whisper; a slow pass slows the
//! captions down but never stops audio from being recorded. One pipeline per channel.

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::thread::sleep;
use std::time::{Duration, Instant};

use meetbutler_engine::incremental::{IncrementalTranscriber, RawSegment, Segment, WindowConfig};
use meetbutler_engine::pipeline::{SessionPipeline, Step};
use meetbutler_engine::{AudioSource, TranscribeError, SAMPLE_RATE};

const SR: usize = SAMPLE_RATE;

#[derive(Default)]
struct FakeStore {
    total: AtomicUsize,
}

impl FakeStore {
    fn grow(&self, seconds: f64) {
        self.total
            .fetch_add((seconds * SR as f64) as usize, Ordering::SeqCst);
    }
}

impl AudioSource for FakeStore {
    fn total_samples(&self) -> usize {
        self.total.load(Ordering::SeqCst)
    }
    fn read(&self, start: usize, end: Option<usize>) -> io::Result<Vec<f32>> {
        let total = self.total_samples();
        let end = end.map_or(total, |e| e.min(total));
        Ok(vec![0.0; end - start.min(end)])
    }
}

/// A fake model that can be slow, scripted, and watched for overlapping calls.
#[derive(Clone, Default)]
struct Whisper {
    delay: Duration,
    calls: Arc<AtomicUsize>,
    concurrent: Arc<AtomicUsize>,
    max_concurrent: Arc<AtomicUsize>,
    script: Arc<Mutex<Vec<Vec<RawSegment>>>>,
}

impl Whisper {
    fn slow(ms: u64) -> Self {
        Self {
            delay: Duration::from_millis(ms),
            ..Self::default()
        }
    }
    fn scripted(lines: &[&[(f64, f64, &str)]]) -> Self {
        let w = Self::default();
        *w.script.lock().unwrap() = lines
            .iter()
            .map(|segs| segs.iter().map(|&(a, b, t)| raw(a, b, t)).collect())
            .collect();
        w
    }
    fn calls(&self) -> usize {
        self.calls.load(Ordering::SeqCst)
    }
    fn func(
        &self,
    ) -> impl FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError> + Send + 'static {
        let me = self.clone();
        move |_| {
            me.calls.fetch_add(1, Ordering::SeqCst);
            let now = me.concurrent.fetch_add(1, Ordering::SeqCst) + 1;
            me.max_concurrent.fetch_max(now, Ordering::SeqCst);
            sleep(me.delay);
            me.concurrent.fetch_sub(1, Ordering::SeqCst);
            let mut script = me.script.lock().unwrap();
            Ok(if script.is_empty() {
                Vec::new()
            } else {
                script.remove(0)
            })
        }
    }
}

fn raw(start: f64, end: f64, text: &str) -> RawSegment {
    RawSegment {
        start,
        end,
        text: text.to_string(),
    }
}

#[derive(Clone, Default)]
struct Collector(Arc<Mutex<Vec<String>>>);

impl Collector {
    fn texts(&self) -> Vec<String> {
        self.0.lock().unwrap().clone()
    }
    fn sink(&self) -> impl FnMut(&[Segment]) -> Result<(), TranscribeError> + Send + 'static {
        let me = self.clone();
        move |segs| {
            me.0.lock()
                .unwrap()
                .extend(segs.iter().map(|s| s.text.clone()));
            Ok(())
        }
    }
}

/// These fakes report speech spanning their whole window, which is what hold-back exists
/// to defer; the wiring tests here are not about window edges.
fn no_holdback() -> WindowConfig {
    WindowConfig {
        holdback_sec: 0.0,
        ..WindowConfig::default()
    }
}

fn build(
    whisper: &Whisper,
    interval_ms: u64,
    cfg: WindowConfig,
) -> (Arc<FakeStore>, SessionPipeline, Collector) {
    let store = Arc::new(FakeStore::default());
    let collector = Collector::default();
    let t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg);
    let pipeline = SessionPipeline::start(t, collector.sink(), Duration::from_millis(interval_ms));
    (store, pipeline, collector)
}

fn wait_until(timeout_ms: u64, mut check: impl FnMut() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    while Instant::now() < deadline {
        if check() {
            return true;
        }
        sleep(Duration::from_millis(5));
    }
    check()
}

#[test]
fn capture_is_never_blocked_by_a_running_pass() {
    let whisper = Whisper::slow(500);
    let (store, mut pipeline, _) = build(&whisper, 20, no_holdback());
    store.grow(1.0);
    assert!(wait_until(2000, || whisper.calls() >= 1));

    let started = Instant::now();
    for _ in 0..200 {
        store.grow(0.01);
    }
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "{:?}",
        started.elapsed()
    );
    pipeline.stop();
}

#[test]
fn the_live_transcript_can_be_read_while_a_pass_is_running() {
    // the UI asks for the transcript at any time; it must not wait for Whisper
    let whisper = Whisper::scripted(&[&[(0.0, 1.0, "first")]]);
    let slow = Whisper {
        delay: Duration::from_millis(400),
        ..whisper.clone()
    };
    let (store, mut pipeline, _) = build(&slow, 20, no_holdback());
    store.grow(1.0);
    assert!(wait_until(2000, || pipeline.transcript().len() == 1));

    store.grow(1.0); // the next pass now takes 400 ms
    assert!(wait_until(2000, || slow.concurrent.load(Ordering::SeqCst) == 1));
    let started = Instant::now();
    let transcript = pipeline.transcript();
    assert!(
        started.elapsed() < Duration::from_millis(100),
        "{:?}",
        started.elapsed()
    );
    assert_eq!(transcript[0].text, "first");
    pipeline.stop();
}

#[test]
fn transcription_passes_never_overlap() {
    let whisper = Whisper::slow(150);
    let (store, mut pipeline, _) = build(&whisper, 10, no_holdback());
    for _ in 0..5 {
        store.grow(1.0);
    }
    sleep(Duration::from_millis(600));
    pipeline.stop();

    assert_eq!(whisper.max_concurrent.load(Ordering::SeqCst), 1);
}

#[test]
fn segments_are_delivered_while_the_session_is_live() {
    let whisper = Whisper::scripted(&[&[(0.0, 1.0, "live caption")]]);
    let (store, mut pipeline, collector) = build(&whisper, 20, no_holdback());
    store.grow(1.0);

    assert!(wait_until(2000, || !collector.texts().is_empty()));
    assert_eq!(collector.texts(), ["live caption"]);
    pipeline.stop();
}

#[test]
fn stop_transcribes_audio_that_arrived_after_the_last_pass() {
    // trailing audio at hangup must not be dropped
    let whisper = Whisper::scripted(&[&[(0.0, 1.0, "last words")]]);
    let (store, mut pipeline, collector) = build(&whisper, 100_000, no_holdback()); // no periodic pass
    store.grow(1.0);

    pipeline.stop();

    assert_eq!(collector.texts(), ["last words"]);
}

#[test]
fn stop_drains_a_backlog_larger_than_one_window() {
    let whisper = Whisper::default();
    let (store, mut pipeline, _) = build(&whisper, 100_000, no_holdback());
    store.grow(150.0); // five windows, no pass yet

    let t = pipeline.stop().expect("the transcriber comes back");

    assert_eq!(t.pending_samples(), 0);
}

/// Records how each pass was asked for; reports backlog until it has been forced.
struct Spy {
    pending: usize,
    forced: Arc<Mutex<Vec<bool>>>,
}

impl Step for Spy {
    fn step(&mut self, force: bool) -> Vec<Segment> {
        self.forced.lock().unwrap().push(force);
        self.pending = self.pending.saturating_sub(30 * SR);
        Vec::new()
    }
    fn pending_samples(&self) -> usize {
        self.pending
    }
    fn transcript(&self) -> &[Segment] {
        &[]
    }
}

#[test]
fn stop_drains_with_forced_passes_only() {
    // once capture has stopped no more audio is coming, so every drain pass is final: a
    // normal pass may decline to commit speech running into the end, and a drain loop of
    // those re-reads the same tail until it hits its pass limit
    let forced = Arc::new(Mutex::new(Vec::new()));
    let spy = Spy {
        pending: 90 * SR,
        forced: forced.clone(),
    };
    let mut pipeline =
        SessionPipeline::start(spy, Collector::default().sink(), Duration::from_secs(100));

    let spy = pipeline.stop().unwrap();

    let forced = forced.lock().unwrap();
    assert!(
        !forced.is_empty() && forced.iter().all(|&f| f),
        "{forced:?}"
    );
    assert_eq!(spy.pending_samples(), 0);
}

#[test]
fn stop_commits_speech_running_into_the_end_without_spinning() {
    // hold-back meets hangup: the held tail is committed by one drain pass
    let calls = Arc::new(AtomicUsize::new(0));
    let still_talking = {
        let calls = calls.clone();
        move |audio: &[f32]| -> Result<Vec<RawSegment>, TranscribeError> {
            calls.fetch_add(1, Ordering::SeqCst);
            Ok(vec![raw(
                0.0,
                audio.len() as f64 / SR as f64,
                "still talking",
            )])
        }
    };
    let store = Arc::new(FakeStore::default());
    let collector = Collector::default();
    let t = IncrementalTranscriber::new(store.clone(), still_talking, WindowConfig::default());
    let mut pipeline = SessionPipeline::start(t, collector.sink(), Duration::from_secs(100));
    store.grow(5.0);

    let t = pipeline.stop().unwrap();

    assert_eq!(collector.texts(), ["still talking"]);
    assert_eq!(calls.load(Ordering::SeqCst), 1);
    assert_eq!(t.pending_samples(), 0);
}

#[test]
fn no_pass_runs_after_stop_returns() {
    let whisper = Whisper::default();
    let (store, mut pipeline, _) = build(&whisper, 10, no_holdback());
    store.grow(1.0);
    pipeline.stop();
    let after_stop = whisper.calls();

    store.grow(5.0);
    sleep(Duration::from_millis(100));

    assert_eq!(whisper.calls(), after_stop);
}

#[test]
fn stop_is_safe_when_nothing_was_ever_captured() {
    let whisper = Whisper::default();
    let (_store, mut pipeline, collector) = build(&whisper, 20, no_holdback());

    pipeline.stop();

    assert!(collector.texts().is_empty());
    assert_eq!(whisper.calls(), 0);
}

#[test]
fn stop_is_idempotent() {
    let whisper = Whisper::default();
    let (_store, mut pipeline, _) = build(&whisper, 20, no_holdback());

    assert!(pipeline.stop().is_some());
    assert!(pipeline.stop().is_none());
}

#[test]
fn dropping_a_pipeline_stops_its_worker() {
    let whisper = Whisper::default();
    let (store, pipeline, _) = build(&whisper, 10, no_holdback());
    drop(pipeline);
    let after_drop = whisper.calls();

    store.grow(5.0);
    sleep(Duration::from_millis(100));

    assert_eq!(whisper.calls(), after_drop);
}

#[test]
fn a_failing_pass_does_not_kill_the_session() {
    // one bad window must not stop later captions from arriving
    let calls = Arc::new(AtomicUsize::new(0));
    let flaky = {
        let calls = calls.clone();
        move |_: &[f32]| -> Result<Vec<RawSegment>, TranscribeError> {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                return Err("bad window".into());
            }
            Ok(vec![raw(0.0, 1.0, "recovered")])
        }
    };
    let store = Arc::new(FakeStore::default());
    let collector = Collector::default();
    let t = IncrementalTranscriber::new(store.clone(), flaky, no_holdback());
    let mut pipeline = SessionPipeline::start(t, collector.sink(), Duration::from_millis(20));

    let recovered = wait_until(3000, || {
        store.grow(1.0);
        collector.texts().contains(&"recovered".to_string())
    });
    pipeline.stop();

    assert!(recovered);
}

#[test]
fn a_failing_or_panicking_sink_does_not_stop_transcription() {
    // a closed panel or a full disk on one write must not take the session down
    let whisper = Whisper::scripted(&[
        &[(0.0, 1.0, "one")],
        &[(0.0, 1.0, "two")],
        &[(0.0, 1.0, "three")],
    ]);
    let store = Arc::new(FakeStore::default());
    let sink_calls = Arc::new(AtomicUsize::new(0));
    let sink = {
        let sink_calls = sink_calls.clone();
        move |_: &[Segment]| -> Result<(), TranscribeError> {
            match sink_calls.fetch_add(1, Ordering::SeqCst) {
                0 => Err("subscriber went away".into()),
                1 => panic!("sink blew up"),
                _ => Ok(()),
            }
        }
    };
    let t = IncrementalTranscriber::new(store.clone(), whisper.func(), no_holdback());
    let mut pipeline = SessionPipeline::start(t, sink, Duration::from_millis(20));

    let done = wait_until(3000, || {
        store.grow(1.0);
        whisper.calls() >= 3
    });
    let t = pipeline.stop().unwrap();

    assert!(done);
    let texts: Vec<&str> = t.transcript().iter().map(|s| s.text.as_str()).collect();
    assert_eq!(texts, ["one", "two", "three"]);
}

#[test]
fn a_panicking_pass_does_not_kill_the_session() {
    let calls = Arc::new(AtomicUsize::new(0));
    let flaky = {
        let calls = calls.clone();
        move |_: &[f32]| -> Result<Vec<RawSegment>, TranscribeError> {
            if calls.fetch_add(1, Ordering::SeqCst) == 0 {
                panic!("decoder blew up");
            }
            Ok(vec![raw(0.0, 1.0, "recovered")])
        }
    };
    let store = Arc::new(FakeStore::default());
    let collector = Collector::default();
    let t = IncrementalTranscriber::new(store.clone(), flaky, no_holdback());
    let mut pipeline = SessionPipeline::start(t, collector.sink(), Duration::from_millis(20));

    let recovered = wait_until(3000, || {
        store.grow(1.0);
        collector.texts().contains(&"recovered".to_string())
    });
    pipeline.stop();

    assert!(recovered);
}

#[test]
fn the_stopped_flag_is_visible_to_the_worker_promptly() {
    // a long interval must not make stop wait for the next tick
    let whisper = Whisper::default();
    let (_store, mut pipeline, _) = build(&whisper, 60_000, no_holdback());
    let started = Instant::now();
    pipeline.stop();
    assert!(
        started.elapsed() < Duration::from_millis(500),
        "{:?}",
        started.elapsed()
    );
}
