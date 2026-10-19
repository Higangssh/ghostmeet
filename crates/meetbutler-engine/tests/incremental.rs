//! IncrementalTranscriber — windowed transcription with absolute timestamps.
//!
//! The invariant that matters: the amount of audio handed to Whisper per pass is bounded
//! by the window, no matter how long the meeting has been running.

use std::io;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};

use meetbutler_engine::incremental::{IncrementalTranscriber, RawSegment, Segment, WindowConfig};
use meetbutler_engine::{AudioSource, TranscribeError, SAMPLE_RATE};

const SR: usize = SAMPLE_RATE;

fn raw(start: f64, end: f64, text: &str) -> RawSegment {
    RawSegment {
        start,
        end,
        text: text.to_string(),
    }
}

/// Duck-types PcmStore without putting hours of audio on disk. Remembers the last
/// window read so a fake model can tell which slice of the session it was handed.
#[derive(Default)]
struct FakeStore {
    total: AtomicUsize,
    last_read: Mutex<(usize, usize)>,
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
        let start = start.min(end);
        *self.last_read.lock().unwrap() = (start, end);
        Ok(vec![0.0; end - start])
    }
}

/// Records every call so tests can assert on how much audio was processed.
#[derive(Clone, Default)]
struct Whisper {
    calls: Arc<Mutex<Vec<usize>>>,
    script: Arc<Mutex<Vec<Vec<RawSegment>>>>,
}

impl Whisper {
    fn scripted(script: Vec<Vec<RawSegment>>) -> Self {
        let w = Self::default();
        w.set_script(script);
        w
    }
    fn set_script(&self, script: Vec<Vec<RawSegment>>) {
        *self.script.lock().unwrap() = script;
    }
    fn calls(&self) -> Vec<usize> {
        self.calls.lock().unwrap().clone()
    }
    fn largest_call(&self) -> usize {
        self.calls().into_iter().max().unwrap_or(0)
    }
    fn func(&self) -> impl FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        let me = self.clone();
        move |audio| {
            me.calls.lock().unwrap().push(audio.len());
            let mut script = me.script.lock().unwrap();
            Ok(if script.is_empty() {
                Vec::new()
            } else {
                script.remove(0)
            })
        }
    }
}

fn cfg(window: f64, overlap: f64) -> WindowConfig {
    WindowConfig {
        window_sec: window,
        overlap_sec: overlap,
        ..WindowConfig::default()
    }
}

fn texts(segs: &[Segment]) -> Vec<&str> {
    segs.iter().map(|s| s.text.as_str()).collect()
}

fn approx(a: f64, b: f64) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn the_defaults_are_the_measured_ones() {
    let d = WindowConfig::default();
    assert_eq!(
        (d.window_sec, d.overlap_sec, d.min_new_sec, d.holdback_sec),
        (30.0, 2.0, 1.0, 1.0)
    );
}

#[test]
fn no_audio_means_no_whisper_call() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store, whisper.func(), WindowConfig::default());

    assert!(t.step(false).is_empty());
    assert!(whisper.calls().is_empty());
}

#[test]
fn skips_whisper_until_enough_new_audio_accumulates() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default());

    store.grow(0.4);
    assert!(t.step(false).is_empty());
    assert!(whisper.calls().is_empty());

    store.grow(0.7); // now 1.1 s total
    t.step(false);
    assert_eq!(whisper.calls().len(), 1);
}

#[test]
fn segment_times_are_absolute_not_window_relative() {
    // a segment found 1 s into a window that starts at 58 s must report 59 s
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    store.grow(60.0);
    for _ in 0..20 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(false);
    }

    whisper.set_script(vec![vec![raw(1.0, 3.0, "second window")]]);
    store.grow(10.0);
    let found = t.step(false);

    assert_eq!(found.len(), 1);
    let window_start = 60.0 - 2.0; // cursor rewound by the overlap
    assert!(approx(found[0].start, window_start + 1.0));
    assert!(approx(found[0].end, window_start + 3.0));
}

#[test]
fn audio_per_pass_stays_bounded_across_a_four_hour_session() {
    // the regression test for re-processing the whole session: cost per pass must not grow
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    for _ in 0..1440 {
        store.grow(10.0); // 4 hours in 10 s ticks
        t.step(false);
    }

    assert_eq!(store.total_samples(), 4 * 3600 * SR);
    assert_eq!(whisper.calls().len(), 1440); // linear in session length
    assert!(whisper.largest_call() <= 30 * SR);
}

#[test]
fn a_stalled_cursor_cannot_grow_the_window_without_bound() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    store.grow(3600.0); // an hour arrives before the first pass
    t.step(false);

    assert!(whisper.largest_call() <= 30 * SR);
}

#[test]
fn a_backlog_is_caught_up_oldest_first_and_never_skipped() {
    // anchoring the window to the newest audio once marked the middle of a backlog as
    // consumed without transcribing it
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    store.grow(90.0);
    t.step(false);
    assert!(
        t.pending_samples() > 0,
        "jumped to the newest audio, skipping the rest"
    );

    for _ in 0..20 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(false);
    }

    assert_eq!(t.pending_samples(), 0);
    assert!(whisper.calls().iter().sum::<usize>() >= 90 * SR); // every second reached Whisper
}

#[test]
fn forced_pass_transcribes_a_trailing_fraction_of_a_second() {
    // a meeting ending mid-word must not drop its final moments
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 0.4, "bye")]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default());

    store.grow(0.4);

    assert!(t.step(false).is_empty()); // below the normal threshold
    assert_eq!(texts(&t.step(true)), ["bye"]);
}

#[test]
fn forced_pass_on_an_empty_session_does_nothing() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store, whisper.func(), WindowConfig::default());

    assert!(t.step(true).is_empty());
    assert!(whisper.calls().is_empty());
}

#[test]
fn overlapping_audio_does_not_produce_duplicate_segments() {
    // speech filling a whole window has no pause to resume from, so the cursor is cut at
    // the window edge and the next pass rewinds by the overlap — without re-emitting
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![
        vec![raw(1.0, 29.5, "hello there")],
        vec![
            raw(0.5, 1.5, "hello there"),
            raw(3.0, 5.0, "brand new line"),
        ],
    ]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    store.grow(30.0);
    let first = t.step(false);
    store.grow(10.0);
    let second = t.step(false);

    assert_eq!(texts(&first), ["hello there"]);
    assert_eq!(texts(&second), ["brand new line"]);
    assert_eq!(texts(t.transcript()), ["hello there", "brand new line"]);
}

#[test]
fn speech_cut_off_at_the_window_edge_is_retried() {
    // trailing audio after the last segment is re-read, so cut words get a second pass
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 5.0, "complete sentence")], vec![]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    store.grow(10.0);
    t.step(false);
    store.grow(1.5);
    t.step(false);

    // the second pass re-reads from the end of the last segment (5 s), not from 10 s
    assert!(whisper.calls()[1] > (1.5 * SR as f64) as usize);
}

#[test]
fn silence_advances_the_cursor_so_it_is_not_rescanned_forever() {
    // when the cursor failed to advance through silence, every later pass hit the window
    // clamp and burned a full 30 s of Whisper work for 5 s of new audio
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    for _ in 0..20 {
        store.grow(5.0);
        t.step(false);
    }

    // every pass sees the new 5 s plus the 2 s overlap — never the whole window
    let last = *whisper.calls().last().unwrap() as f64;
    assert!((last / (7.0 * SR as f64) - 1.0).abs() < 0.01, "{last}");
    assert!(whisper.largest_call() < 30 * SR);
}

#[test]
fn segments_expose_a_speaker_slot_for_diarization() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 1.0, "who said this")]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default());

    store.grow(5.0);
    let segment = t.step(false).remove(0);

    assert_eq!(segment.speaker, None);
    let json = serde_json::to_value(&segment).unwrap();
    assert!(json.as_object().unwrap().contains_key("speaker"));
}

#[test]
fn a_transcriber_can_label_its_channel_as_the_speaker() {
    // the microphone is "me" and system audio is "them" until diarization splits them
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 1.0, "hello")]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default())
        .with_speaker("me");

    store.grow(5.0);
    assert_eq!(t.step(false)[0].speaker.as_deref(), Some("me"));
}

#[test]
fn transcript_and_full_text_preserve_order() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![
        vec![raw(0.0, 1.0, "first")],
        vec![raw(0.0, 1.0, "second")],
    ]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 0.0));

    store.grow(10.0);
    t.step(false);
    store.grow(10.0);
    t.step(false);

    assert_eq!(t.full_text(), "first second");
    assert_eq!(texts(t.transcript()), ["first", "second"]);
}

#[test]
fn blank_segments_from_whisper_are_dropped() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 1.0, "   "), raw(1.0, 2.0, "real text")]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default());

    store.grow(5.0);
    assert_eq!(texts(&t.step(false)), ["real text"]);
}

#[test]
fn text_is_trimmed() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::scripted(vec![vec![raw(0.0, 1.0, "  padded  ")]]);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), WindowConfig::default());

    store.grow(5.0);
    assert_eq!(texts(&t.step(false)), ["padded"]);
}

#[test]
fn timestamps_remain_accurate_after_four_hours() {
    // absolute time comes from the sample cursor, so long sessions do not drift
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    for _ in 0..1439 {
        store.grow(10.0);
        t.step(false);
    }

    whisper.set_script(vec![vec![raw(9.0, 9.5, "final word")]]);
    store.grow(10.0);
    let found = t.step(false);

    let duration = store.total_samples() as f64 / SR as f64;
    assert_eq!(duration, 4.0 * 3600.0);
    assert!(found[0].end <= duration);
    assert!(found[0].end > duration - 30.0);
}

#[test]
fn whisper_failure_does_not_lose_the_session() {
    // a bad pass must not kill an in-progress meeting or stall the cursor
    let store = Arc::new(FakeStore::default());
    let exploding =
        |_: &[f32]| -> Result<Vec<RawSegment>, TranscribeError> { Err("model blew up".into()) };
    let mut t = IncrementalTranscriber::new(store.clone(), exploding, WindowConfig::default());
    store.grow(10.0);

    assert!(t.step(false).is_empty());
    assert!(t.transcript().is_empty());
    assert_eq!(
        t.pending_samples(),
        0,
        "the failed window is consumed, not retried forever"
    );
}

#[test]
fn pending_samples_reports_audio_not_yet_consumed() {
    // used at session end to drain trailing audio instead of dropping it
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));

    assert_eq!(t.pending_samples(), 0);

    store.grow(45.0);
    assert_eq!(t.pending_samples(), 45 * SR);

    t.step(false); // one pass can only consume one window

    assert!(t.pending_samples() > 0);
    assert!(t.pending_samples() < 45 * SR);
}

#[test]
fn repeated_steps_drain_a_long_backlog_to_zero() {
    let store = Arc::new(FakeStore::default());
    let whisper = Whisper::default();
    let mut t = IncrementalTranscriber::new(store.clone(), whisper.func(), cfg(30.0, 2.0));
    store.grow(600.0);

    for _ in 0..100 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(false);
    }

    assert_eq!(t.pending_samples(), 0);
    assert!(whisper.largest_call() <= 30 * SR);
}

#[test]
fn a_segment_serialises_with_the_fields_the_ui_uses() {
    let segment = Segment::new("hi", 1.0, 2.0);

    let data = serde_json::to_value(&segment).unwrap();

    assert_eq!(data["text"], "hi");
    assert_eq!(data["start"], 1.0);
    assert_eq!(data["end"], 2.0);
    assert!(
        data["timestamp"].as_f64().unwrap() > 1.7e9,
        "wall-clock seconds"
    );
}

// --- window boundaries ---------------------------------------------------------------
//
// On Korean read speech the live path (10 s passes) once scored CER ~0.25-0.31 against
// ~0.02-0.06 per utterance, on every engine, because speech was duplicated at window
// boundaries: a segment cut off by the window edge was committed as final, and the next
// window re-read the overlap and committed the same speech again.

/// Transcribes a fixed script of utterances the way Whisper does at window edges.
///
/// Speech running past the end of the window comes back cut ("text~cut"); speech that
/// began before the window comes back as a fragment ("~text"). A correct transcript
/// contains neither marker and each utterance exactly once.
fn timeline_whisper(
    store: Arc<FakeStore>,
    utterances: Vec<(f64, f64, String)>,
    calls: Arc<Mutex<Vec<usize>>>,
) -> impl FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
    move |audio| {
        calls.lock().unwrap().push(audio.len());
        let (a, b) = *store.last_read.lock().unwrap();
        let (ws, we) = (a as f64 / SR as f64, b as f64 / SR as f64);
        let mut out = Vec::new();
        for (start, end, text) in &utterances {
            if *end <= ws || *start >= we {
                continue;
            }
            let label = format!(
                "{}{}{}",
                if *start < ws { "~" } else { "" },
                text,
                if *end > we { "~cut" } else { "" }
            );
            out.push(raw(start.max(ws) - ws, end.min(we) - ws, &label));
        }
        Ok(out)
    }
}

fn script(items: &[(f64, f64, &str)]) -> Vec<(f64, f64, String)> {
    items
        .iter()
        .map(|&(a, b, t)| (a, b, t.to_string()))
        .collect()
}

fn run_live<S, T>(store: &FakeStore, t: &mut IncrementalTranscriber<S, T>, seconds: f64)
where
    S: AudioSource,
    T: FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError>,
{
    for _ in 0..(seconds / 10.0) as usize {
        store.grow(10.0);
        t.step(false);
    }
    for _ in 0..10 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(true);
    }
}

#[test]
fn speech_cut_by_the_window_edge_is_not_committed_twice() {
    // live passes: a sentence still being spoken at the edge waits for the next pass
    let items = [
        (1.0, 6.0, "one"),
        (7.0, 13.0, "two"),
        (14.0, 18.5, "three"),
        (21.0, 28.0, "four"),
    ];
    let store = Arc::new(FakeStore::default());
    let calls = Arc::default();
    let whisper = timeline_whisper(store.clone(), script(&items), calls);
    let mut t = IncrementalTranscriber::new(store.clone(), whisper, cfg(30.0, 2.0));

    run_live(&store, &mut t, 30.0);

    assert_eq!(texts(t.transcript()), ["one", "two", "three", "four"]);
    let times: Vec<(f64, f64)> = t.transcript().iter().map(|s| (s.start, s.end)).collect();
    let want: Vec<(f64, f64)> = items.iter().map(|&(a, b, _)| (a, b)).collect();
    assert_eq!(times, want);
}

#[test]
fn a_long_segment_spanning_the_overlap_is_not_re_emitted() {
    // backlog passes: the sentence crossing the 30 s edge appears once, whole
    let items = [
        (2.0, 12.0, "a"),
        (14.0, 26.0, "b"),
        (27.0, 35.0, "c"),
        (37.0, 50.0, "d"),
    ];
    let store = Arc::new(FakeStore::default());
    let whisper = timeline_whisper(store.clone(), script(&items), Arc::default());
    let mut t = IncrementalTranscriber::new(store.clone(), whisper, cfg(30.0, 2.0));

    store.grow(60.0);
    for _ in 0..10 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(true);
    }

    assert_eq!(texts(t.transcript()), ["a", "b", "c", "d"]);
    assert_eq!(t.pending_samples(), 0);
}

#[test]
fn a_held_back_tail_is_committed_by_the_final_forced_pass() {
    // holding back the edge must never lose it: the session's last words still land
    let store = Arc::new(FakeStore::default());
    let whisper = timeline_whisper(
        store.clone(),
        script(&[(1.0, 5.0, "x"), (6.0, 9.8, "y")]),
        Arc::default(),
    );
    let mut t = IncrementalTranscriber::new(store.clone(), whisper, cfg(30.0, 2.0));

    store.grow(10.0);
    assert_eq!(texts(&t.step(false)), ["x"]); // y runs into the edge
    assert_eq!(texts(&t.step(true)), ["y"]);
    assert_eq!(t.pending_samples(), 0);
}

#[test]
fn speech_longer_than_a_window_still_makes_progress_within_bounds() {
    // holding back must not stall a cursor when no pause ever comes
    let store = Arc::new(FakeStore::default());
    let calls: Arc<Mutex<Vec<usize>>> = Arc::default();
    let whisper = timeline_whisper(
        store.clone(),
        script(&[(0.0, 100.0, "monologue")]),
        calls.clone(),
    );
    let mut t = IncrementalTranscriber::new(store.clone(), whisper, cfg(30.0, 2.0));

    store.grow(100.0);
    for _ in 0..20 {
        if t.pending_samples() == 0 {
            break;
        }
        t.step(true);
    }

    assert_eq!(t.pending_samples(), 0);
    assert!(
        !t.transcript().is_empty(),
        "the monologue was dropped instead of committed"
    );
    assert!(calls.lock().unwrap().iter().all(|&c| c <= 30 * SR));
}

#[test]
fn holding_back_keeps_the_per_pass_window_bounded_for_four_hours() {
    // speech that always runs into the edge must not grow the window past its cap
    let items: Vec<(f64, f64, String)> = (0..4 * 3600)
        .step_by(10)
        .enumerate()
        .map(|(i, t)| (t as f64 + 1.0, t as f64 + 10.0, format!("s{i}")))
        .collect();
    let n = items.len();
    let store = Arc::new(FakeStore::default());
    let calls: Arc<Mutex<Vec<usize>>> = Arc::default();
    let whisper = timeline_whisper(store.clone(), items, calls.clone());
    let mut t = IncrementalTranscriber::new(store.clone(), whisper, cfg(30.0, 2.0));

    run_live(&store, &mut t, 4.0 * 3600.0);

    assert!(calls.lock().unwrap().iter().all(|&c| c <= 30 * SR));
    assert_eq!(t.transcript().len(), n);
    assert!(t.transcript().iter().all(|s| !s.text.contains('~')));
}

#[test]
fn a_panicking_model_is_a_failed_pass_not_a_stuck_window() {
    // a window that crashes the decoder every time must be skipped, not retried forever
    let store = Arc::new(FakeStore::default());
    let panicking =
        |_: &[f32]| -> Result<Vec<RawSegment>, TranscribeError> { panic!("decoder blew up") };
    let mut t = IncrementalTranscriber::new(store.clone(), panicking, WindowConfig::default());
    store.grow(10.0);

    assert!(t.step(false).is_empty());
    assert_eq!(t.pending_samples(), 0);
}
