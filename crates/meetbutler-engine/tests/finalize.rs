//! After the meeting: the final pass replaces the live transcript, whole, or not at all.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Arc;
use std::time::Duration;

use meetbutler_engine::incremental::RawSegment;
use meetbutler_engine::recorder::{Channel, Recorder, RecorderConfig, RecorderError};
use meetbutler_engine::span_decoder::{SpanDecode, SpeechProbs};
use meetbutler_engine::store::{SessionStatus, TranscriptKind};
use meetbutler_engine::{TranscribeError, SAMPLE_RATE};

/// Live captions: one fixed line per pass.
fn live(
    _: Channel,
) -> impl FnMut(&[f32]) -> Result<Vec<RawSegment>, TranscribeError> + Send + 'static {
    |_| {
        Ok(vec![RawSegment {
            start: 0.0,
            end: 0.5,
            text: "provisional".into(),
        }])
    }
}

struct LoudnessVad;

impl SpeechProbs for LoudnessVad {
    fn probs(&mut self, audio: &[f32]) -> Result<Vec<f32>, TranscribeError> {
        Ok(audio
            .chunks_exact(512)
            .map(|f| if f[0] != 0.0 { 0.9 } else { 0.0 })
            .collect())
    }
}

/// The final model: names the channel it heard.
struct Final(&'static str);

impl SpanDecode for Final {
    fn decode(&mut self, audio: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        Ok(vec![RawSegment {
            start: 0.0,
            end: audio.len() as f64 / SAMPLE_RATE as f64,
            text: self.0.into(),
        }])
    }
}

struct Broken;

impl SpanDecode for Broken {
    fn decode(&mut self, _: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        Err("out of memory".into())
    }
}

fn final_for(c: Channel) -> Final {
    Final(match c {
        Channel::Mic => "final mine",
        Channel::System => "final theirs",
    })
}

fn speech(secs: f32) -> Vec<f32> {
    vec![0.3; (secs * SAMPLE_RATE as f32) as usize]
}

fn silence(secs: f32) -> Vec<f32> {
    vec![0.0; (secs * SAMPLE_RATE as f32) as usize]
}

fn cfg() -> RecorderConfig {
    RecorderConfig {
        interval: Duration::from_secs(100),
        ..RecorderConfig::default()
    }
}

/// A stopped call: the mic speaks at 1-3 s, the other side at 5-7 s.
fn recorded(dir: &tempfile::TempDir) -> Recorder {
    let r = Recorder::open(dir.path(), cfg()).unwrap();
    let audio = r
        .start(
            "call",
            &[Channel::Mic, Channel::System],
            Some("ko".into()),
            live,
        )
        .unwrap();
    audio[0]
        .append_f32(&[silence(1.0), speech(2.0), silence(6.0)].concat())
        .unwrap();
    audio[1]
        .append_f32(&[silence(5.0), speech(2.0), silence(2.0)].concat())
        .unwrap();
    r.stop("call").unwrap();
    r
}

fn texts(r: &Recorder, id: &str) -> Vec<String> {
    r.transcript(id)
        .unwrap()
        .unwrap()
        .into_iter()
        .map(|s| s.text)
        .collect()
}

#[test]
fn the_final_transcript_replaces_the_live_one_per_channel_in_time_order() {
    let dir = tempfile::tempdir().unwrap();
    let r = recorded(&dir);
    assert!(texts(&r, "call").iter().all(|t| t == "provisional"));

    let session = r
        .finalize(
            "call",
            || Ok(LoudnessVad),
            |c| Ok(final_for(c)),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();

    assert_eq!(session.transcript, TranscriptKind::Final);
    assert_eq!(session.status, SessionStatus::Stopped);
    let segs = r.transcript("call").unwrap().unwrap();
    let got: Vec<(&str, &str)> = segs
        .iter()
        .map(|s| (s.text.as_str(), s.speaker.as_deref().unwrap()))
        .collect();
    assert_eq!(got, [("final mine", "me"), ("final theirs", "them")]);
    assert!(segs[0].start > 0.5 && segs[1].start > 4.5);
    assert_eq!(session.transcript_segments, 2);
}

#[test]
fn a_failed_final_pass_keeps_the_live_transcript_and_says_why() {
    let dir = tempfile::tempdir().unwrap();
    let r = recorded(&dir);
    let before = texts(&r, "call");

    let err = r.finalize(
        "call",
        || Ok(LoudnessVad),
        |_| Ok(Broken),
        &AtomicBool::new(false),
        |_| {},
    );

    assert!(err.is_err());
    assert_eq!(texts(&r, "call"), before);
    let s = r.session("call").unwrap().unwrap();
    assert_eq!(
        (s.status, s.transcript),
        (SessionStatus::Stopped, TranscriptKind::Live)
    );
    assert!(s.error.unwrap().contains("out of memory"));
}

#[test]
fn a_cancelled_final_pass_changes_nothing() {
    // quitting the app mid-pass: the live transcript stays, the pass can run again later
    let dir = tempfile::tempdir().unwrap();
    let r = recorded(&dir);
    let cancel = AtomicBool::new(true);

    assert!(matches!(
        r.finalize(
            "call",
            || Ok(LoudnessVad),
            |c| Ok(final_for(c)),
            &cancel,
            |_| {}
        ),
        Err(RecorderError::Cancelled)
    ));
    let s = r.session("call").unwrap().unwrap();
    assert_eq!(
        (s.status, s.transcript, s.error),
        (SessionStatus::Stopped, TranscriptKind::Live, None)
    );

    cancel.store(false, Ordering::SeqCst);
    assert!(r
        .finalize(
            "call",
            || Ok(LoudnessVad),
            |c| Ok(final_for(c)),
            &cancel,
            |_| {}
        )
        .is_ok());
}

#[test]
fn a_session_still_recording_cannot_be_finalized() {
    let dir = tempfile::tempdir().unwrap();
    let r = Recorder::open(dir.path(), cfg()).unwrap();
    r.start("live", &[Channel::Mic], None, live).unwrap();

    let err = r.finalize(
        "live",
        || Ok(LoudnessVad),
        |c| Ok(final_for(c)),
        &AtomicBool::new(false),
        |_| {},
    );

    assert!(matches!(err, Err(RecorderError::AlreadyRecording)));
    r.stop("live").unwrap();
}

#[test]
fn an_unknown_session_cannot_be_finalized() {
    let dir = tempfile::tempdir().unwrap();
    let r = Recorder::open(dir.path(), cfg()).unwrap();
    assert!(matches!(
        r.finalize(
            "nope",
            || Ok(LoudnessVad),
            |c| Ok(final_for(c)),
            &AtomicBool::new(false),
            |_| {}
        ),
        Err(RecorderError::NotFound)
    ));
}

#[test]
fn an_interrupted_session_can_be_finalized_from_its_audio() {
    // a crash mid-meeting still ends with a complete final transcript
    let dir = tempfile::tempdir().unwrap();
    {
        let r = Recorder::open(dir.path(), cfg()).unwrap();
        let audio = r.start("crashed", &[Channel::Mic], None, live).unwrap();
        audio[0]
            .append_f32(&[silence(1.0), speech(2.0), silence(2.0)].concat())
            .unwrap();
        std::mem::forget(r);
    }
    let r = Recorder::open(dir.path(), cfg()).unwrap();

    let s = r
        .finalize(
            "crashed",
            || Ok(LoudnessVad),
            |c| Ok(final_for(c)),
            &AtomicBool::new(false),
            |_| {},
        )
        .unwrap();

    assert_eq!(s.transcript, TranscriptKind::Final);
    assert_eq!(s.status, SessionStatus::Stopped);
    assert!((s.duration_sec - 5.0).abs() < 1e-6, "{}", s.duration_sec);
    assert_eq!(texts(&r, "crashed"), ["final mine"]);
}

#[test]
fn progress_covers_every_channel_and_ends_at_one() {
    let dir = tempfile::tempdir().unwrap();
    let r = recorded(&dir);
    let seen = Arc::new(std::sync::Mutex::new(Vec::new()));

    r.finalize(
        "call",
        || Ok(LoudnessVad),
        |c| Ok(final_for(c)),
        &AtomicBool::new(false),
        {
            let seen = seen.clone();
            move |p| seen.lock().unwrap().push(p)
        },
    )
    .unwrap();

    let seen = seen.lock().unwrap();
    assert!(seen.windows(2).all(|w| w[0] <= w[1]), "{seen:?}");
    assert!(seen.iter().any(|&p| p > 0.0 && p < 1.0));
    assert_eq!(*seen.last().unwrap(), 1.0);
}

#[test]
fn a_model_that_fails_to_load_fails_the_pass_cleanly() {
    let dir = tempfile::tempdir().unwrap();
    let r = recorded(&dir);

    let err = r.finalize(
        "call",
        || Ok(LoudnessVad),
        |_| Err::<Final, _>("no such model".into()),
        &AtomicBool::new(false),
        |_| {},
    );

    assert!(err.is_err());
    assert_eq!(
        r.session("call").unwrap().unwrap().transcript,
        TranscriptKind::Live
    );
}
