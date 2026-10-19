//! Recorder — a meeting from start to stop, and what is left of it after a restart.
//!
//! Opening a Recorder on an existing data folder is a genuine restart: nothing in memory
//! carries over, everything has to come back from disk.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use meetbutler_engine::incremental::{RawSegment, Segment};
use meetbutler_engine::recorder::{Channel, Recorder, RecorderConfig, StartError};
use meetbutler_engine::store::{SessionStatus, TranscriptKind};
use meetbutler_engine::{AudioSource, TranscribeError, SAMPLE_RATE};

/// Says the channel's line once per pass that has audio, ending at a pause.
fn stub(
    channel: Channel,
) -> impl FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError> + Send + 'static {
    let text = match channel {
        Channel::Mic => "we agreed to ship on friday",
        Channel::System => "sounds good to me",
    };
    move |_| {
        Ok(vec![RawSegment {
            start: 0.0,
            end: 0.5,
            text: text.into(),
        }])
    }
}

fn config(interval_ms: u64) -> RecorderConfig {
    RecorderConfig {
        interval: Duration::from_millis(interval_ms),
        ..RecorderConfig::default()
    }
}

fn one_second() -> Vec<f32> {
    vec![0.1; SAMPLE_RATE]
}

/// Enough audio that the stub's segment ends well before the hold-back margin, so a
/// live pass commits it.
fn three_seconds() -> Vec<f32> {
    vec![0.1; 3 * SAMPLE_RATE]
}

fn texts(segs: &[Segment]) -> Vec<&str> {
    segs.iter().map(|s| s.text.as_str()).collect()
}

fn wait_until<T>(timeout_ms: u64, mut check: impl FnMut() -> Option<T>) -> Option<T> {
    let deadline = Instant::now() + Duration::from_millis(timeout_ms);
    loop {
        if let Some(v) = check() {
            return Some(v);
        }
        if Instant::now() >= deadline {
            return None;
        }
        std::thread::sleep(Duration::from_millis(10));
    }
}

/// Record `seconds` of mic audio into a session and stop it.
fn record(dir: &tempfile::TempDir, id: &str, seconds: usize) {
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    let audio = recorder
        .start(id, &[Channel::Mic], Some("en".into()), stub)
        .unwrap();
    for _ in 0..seconds {
        audio[0].append_f32(&one_second()).unwrap();
    }
    recorder.stop(id).unwrap();
}

#[test]
fn transcript_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    let before = {
        let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
        let audio = recorder
            .start("persisted", &[Channel::Mic], None, stub)
            .unwrap();
        audio[0].append_f32(&one_second()).unwrap();
        recorder.stop("persisted").unwrap();
        recorder.transcript("persisted").unwrap().unwrap()
    };

    let after = Recorder::open(dir.path(), config(100_000))
        .unwrap()
        .transcript("persisted")
        .unwrap()
        .unwrap();

    assert!(!before.is_empty());
    assert_eq!(after, before);
    assert!(texts(&after).join(" ").contains("ship on friday"));
}

#[test]
fn session_list_survives_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    record(&dir, "persisted", 2);

    let sessions = Recorder::open(dir.path(), config(100_000))
        .unwrap()
        .sessions()
        .unwrap();

    assert_eq!(sessions.len(), 1);
    let s = &sessions[0];
    assert_eq!(s.session_id, "persisted");
    assert_eq!(s.status, SessionStatus::Stopped);
    assert_eq!(s.sources, ["mic"]);
    assert_eq!(s.language.as_deref(), Some("en"));
    assert_eq!(s.transcript, TranscriptKind::Live);
    assert!((s.duration_sec - 2.0).abs() < 1e-6, "{}", s.duration_sec);
    assert!(s.stopped_at.unwrap() >= s.started_at);
    assert_eq!(s.transcript_segments, 1);
}

#[test]
fn segments_are_stored_with_their_channel_as_speaker() {
    let dir = tempfile::tempdir().unwrap();
    record(&dir, "persisted", 1);

    let segments = Recorder::open(dir.path(), config(100_000))
        .unwrap()
        .transcript("persisted")
        .unwrap()
        .unwrap();

    assert_eq!(segments[0].speaker.as_deref(), Some("me"));
    assert!(segments[0].end > segments[0].start);
}

#[test]
fn the_audio_stays_on_disk_after_stop() {
    // it is the recording, and the after-meeting pass reads it again
    let dir = tempfile::tempdir().unwrap();
    record(&dir, "persisted", 2);

    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    let audio = recorder.audio("persisted", Channel::Mic).unwrap().unwrap();
    assert_eq!(audio.total_samples(), 2 * SAMPLE_RATE);
}

#[test]
fn segments_are_written_during_the_meeting_not_only_at_the_end() {
    // a crash four hours in should not cost the whole transcript
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(50)).unwrap();
    let audio = recorder.start("live", &[Channel::Mic], None, stub).unwrap();
    audio[0].append_f32(&three_seconds()).unwrap();

    let stored = wait_until(5000, || {
        let segs = recorder.store().load_segments("live").unwrap();
        (!segs.is_empty()).then_some(segs)
    });

    assert!(
        stored.is_some(),
        "nothing was persisted while the session was still open"
    );
    recorder.stop("live").unwrap();
}

#[test]
fn a_live_session_is_served_from_the_running_pipeline() {
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(50)).unwrap();
    let audio = recorder.start("live", &[Channel::Mic], None, stub).unwrap();
    audio[0].append_f32(&three_seconds()).unwrap();

    let count = wait_until(5000, || {
        let n = recorder.transcript("live").unwrap().unwrap().len();
        (n > 0).then_some(n)
    });

    assert!(count.is_some());
    assert_eq!(
        recorder.session("live").unwrap().unwrap().status,
        SessionStatus::Recording
    );
    recorder.stop("live").unwrap();
}

#[test]
fn unknown_session_has_no_transcript_even_after_a_restart() {
    let dir = tempfile::tempdir().unwrap();
    record(&dir, "real", 1);

    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    assert_eq!(recorder.transcript("never-existed").unwrap(), None);
    assert_eq!(recorder.session("never-existed").unwrap(), None);
    assert!(recorder.stop("never-existed").is_err());
}

#[test]
fn both_channels_are_transcribed_separately_and_merged_in_time_order() {
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    let audio = recorder
        .start("call", &[Channel::Mic, Channel::System], None, stub)
        .unwrap();
    audio[0].append_f32(&one_second()).unwrap();
    audio[1].append_f32(&one_second()).unwrap();
    audio[1].append_f32(&one_second()).unwrap();

    let session = recorder.stop("call").unwrap();
    let transcript = recorder.transcript("call").unwrap().unwrap();

    assert_eq!(session.sources, ["mic", "system"]);
    assert!(
        (session.duration_sec - 2.0).abs() < 1e-6,
        "the longer channel: {}",
        session.duration_sec
    );
    let speakers: Vec<_> = transcript
        .iter()
        .map(|s| s.speaker.as_deref().unwrap())
        .collect();
    assert!(
        speakers.contains(&"me") && speakers.contains(&"them"),
        "{speakers:?}"
    );
    assert!(transcript.windows(2).all(|w| w[0].start <= w[1].start));
}

#[test]
fn a_session_already_recording_cannot_be_started_again() {
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    recorder.start("dup", &[Channel::Mic], None, stub).unwrap();

    let again = recorder.start("dup", &[Channel::Mic], None, stub);

    assert!(matches!(again, Err(StartError::AlreadyRecording)));
    recorder.stop("dup").unwrap();
}

#[test]
fn a_finished_session_id_is_not_reused() {
    // restarting would append to the old recording and corrupt both
    let dir = tempfile::tempdir().unwrap();
    record(&dir, "done", 1);

    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    assert!(matches!(
        recorder.start("done", &[Channel::Mic], None, stub),
        Err(StartError::Exists)
    ));
}

#[test]
fn a_session_needs_at_least_one_channel() {
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    assert!(matches!(
        recorder.start("none", &[], None, stub),
        Err(StartError::NoChannels)
    ));
}

#[test]
fn session_ids_cannot_escape_the_data_folder() {
    let dir = tempfile::tempdir().unwrap();
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    for id in ["../evil", "a/b", "", ".", "..", "c:\\x"] {
        assert!(
            matches!(
                recorder.start(id, &[Channel::Mic], None, stub),
                Err(StartError::BadId)
            ),
            "{id}"
        );
    }
}

#[test]
fn a_crash_mid_meeting_keeps_the_audio_and_the_transcript_so_far() {
    let dir = tempfile::tempdir().unwrap();
    {
        let recorder = Recorder::open(dir.path(), config(50)).unwrap();
        let audio = recorder
            .start("crashed", &[Channel::Mic], None, stub)
            .unwrap();
        audio[0].append_f32(&three_seconds()).unwrap();
        wait_until(5000, || {
            (!recorder
                .store()
                .load_segments("crashed")
                .unwrap()
                .is_empty())
            .then_some(())
        })
        .expect("a pass ran");
        // the process dies: no stop, no drain, nothing runs on the way out
        std::mem::forget(recorder);
    }

    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    let session = recorder.session("crashed").unwrap().unwrap();
    assert_eq!(session.status, SessionStatus::Interrupted);
    assert!(!recorder.transcript("crashed").unwrap().unwrap().is_empty());
    assert_eq!(
        recorder
            .audio("crashed", Channel::Mic)
            .unwrap()
            .unwrap()
            .total_samples(),
        3 * SAMPLE_RATE
    );
}

#[test]
fn new_segments_are_announced_with_their_session() {
    let dir = tempfile::tempdir().unwrap();
    let seen: Arc<Mutex<Vec<(String, usize)>>> = Arc::default();
    let cfg = RecorderConfig {
        on_segments: Some({
            let seen = seen.clone();
            Arc::new(move |id: &str, segs: &[Segment]| {
                seen.lock().unwrap().push((id.to_string(), segs.len()))
            })
        }),
        ..config(100_000)
    };
    let recorder = Recorder::open(dir.path(), cfg).unwrap();
    let audio = recorder.start("told", &[Channel::Mic], None, stub).unwrap();
    audio[0].append_f32(&one_second()).unwrap();

    recorder.stop("told").unwrap();

    assert_eq!(*seen.lock().unwrap(), [("told".to_string(), 1)]);
}

#[test]
fn audio_arriving_after_stop_is_dropped() {
    // the capture thread can still be draining when the session ends
    let dir = tempfile::tempdir().unwrap();
    let calls = Arc::new(AtomicUsize::new(0));
    let recorder = Recorder::open(dir.path(), config(100_000)).unwrap();
    let counting = {
        let calls = calls.clone();
        move |c| {
            let calls = calls.clone();
            let mut inner = stub(c);
            move |a: &[f32]| {
                calls.fetch_add(1, Ordering::SeqCst);
                inner(a)
            }
        }
    };
    let audio = recorder
        .start("late", &[Channel::Mic], None, counting)
        .unwrap();
    audio[0].append_f32(&one_second()).unwrap();
    recorder.stop("late").unwrap();

    audio[0].append_f32(&one_second()).unwrap();

    assert_eq!(audio[0].total_samples(), SAMPLE_RATE);
}
