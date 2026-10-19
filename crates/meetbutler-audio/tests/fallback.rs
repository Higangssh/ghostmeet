//! Opening sources: a meeting records with whatever can be opened.
//!
//! System audio on macOS needs the screen-recording permission. Without it (or on a
//! platform not wired up yet) recording still starts, from the microphone alone, and the
//! caller is told what was left out (requirements §12).

use std::sync::{Arc, Mutex};

use meetbutler_audio::{open_sources, AudioSink, Capture, CaptureError, Source};

struct Fake {
    stopped: Arc<Mutex<Vec<Source>>>,
    source: Source,
}

impl Capture for Fake {
    fn stop(&mut self) {
        self.stopped.lock().unwrap().push(self.source);
    }
}

/// A backend where `denied` sources fail to open.
fn backend(
    denied: &'static [Source],
    stopped: Arc<Mutex<Vec<Source>>>,
) -> impl FnMut(Source, AudioSink) -> Result<Box<dyn Capture>, CaptureError> {
    move |source, _sink| {
        if denied.contains(&source) {
            return Err(match source {
                Source::System => CaptureError::PermissionDenied("screen recording".into()),
                Source::Mic => CaptureError::NoDevice,
            });
        }
        Ok(Box::new(Fake {
            stopped: stopped.clone(),
            source,
        }))
    }
}

fn sink(_: Source) -> AudioSink {
    Box::new(|_: &[f32]| {})
}

#[test]
fn both_sources_open_when_allowed() {
    let opened = open_sources(
        &[Source::Mic, Source::System],
        sink,
        backend(&[], Arc::default()),
    )
    .unwrap();
    assert_eq!(opened.sources(), [Source::Mic, Source::System]);
    assert!(opened.failed.is_empty());
}

#[test]
fn without_the_system_audio_permission_the_mic_still_records() {
    let opened = open_sources(
        &[Source::Mic, Source::System],
        sink,
        backend(&[Source::System], Arc::default()),
    )
    .unwrap();

    assert_eq!(opened.sources(), [Source::Mic]);
    assert_eq!(opened.failed.len(), 1);
    assert_eq!(opened.failed[0].0, Source::System);
    assert!(matches!(
        opened.failed[0].1,
        CaptureError::PermissionDenied(_)
    ));
}

#[test]
fn system_audio_alone_still_records_without_a_mic() {
    let opened = open_sources(
        &[Source::Mic, Source::System],
        sink,
        backend(&[Source::Mic], Arc::default()),
    )
    .unwrap();
    assert_eq!(opened.sources(), [Source::System]);
}

#[test]
fn nothing_opened_is_an_error_naming_every_failure() {
    let err = open_sources(
        &[Source::Mic, Source::System],
        sink,
        backend(&[Source::Mic, Source::System], Arc::default()),
    )
    .err()
    .unwrap();
    assert_eq!(err.len(), 2);
}

#[test]
fn asking_for_nothing_is_an_error() {
    assert!(open_sources(&[], sink, backend(&[], Arc::default())).is_err());
}

#[test]
fn stopping_stops_every_opened_source() {
    let stopped = Arc::new(Mutex::new(Vec::new()));
    let mut opened = open_sources(
        &[Source::Mic, Source::System],
        sink,
        backend(&[], stopped.clone()),
    )
    .unwrap();

    opened.stop();
    opened.stop(); // idempotent

    assert_eq!(*stopped.lock().unwrap(), [Source::Mic, Source::System]);
}

#[test]
fn dropping_stops_the_sources_too() {
    let stopped = Arc::new(Mutex::new(Vec::new()));
    drop(open_sources(&[Source::Mic], sink, backend(&[], stopped.clone())).unwrap());
    assert_eq!(*stopped.lock().unwrap(), [Source::Mic]);
}

#[test]
fn a_source_asked_for_twice_opens_once() {
    let opened = open_sources(
        &[Source::Mic, Source::Mic],
        sink,
        backend(&[], Arc::default()),
    )
    .unwrap();
    assert_eq!(opened.sources(), [Source::Mic]);
}
