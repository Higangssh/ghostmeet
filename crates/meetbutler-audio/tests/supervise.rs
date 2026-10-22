//! A capture whose stream can end under it (the default device changes, a headset is
//! unplugged) is built again on its thread, and keeps going until stopped.

use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

use meetbutler_audio::supervise::{supervised, Rebuild};
use meetbutler_audio::CaptureError;

/// Counts how many were alive and how many were built.
#[derive(Clone, Default)]
struct Tally {
    built: Arc<AtomicUsize>,
    alive: Arc<AtomicUsize>,
    // the rebuild handle each build was given, to pull from "the callback"
    handles: Arc<Mutex<Vec<Rebuild>>>,
}

struct Stream(Tally);

impl Drop for Stream {
    fn drop(&mut self) {
        self.0.alive.fetch_sub(1, Ordering::SeqCst);
    }
}

fn until(cond: impl Fn() -> bool) -> bool {
    let deadline = Instant::now() + Duration::from_secs(3);
    while Instant::now() < deadline {
        if cond() {
            return true;
        }
        std::thread::sleep(Duration::from_millis(5));
    }
    cond()
}

const RETRY: Duration = Duration::from_millis(20);

#[test]
fn a_first_build_that_fails_fails_the_open() {
    let r = supervised("t", RETRY, |_| -> Result<Stream, _> {
        Err(CaptureError::NoDevice)
    });
    assert!(matches!(r, Err(CaptureError::NoDevice)));
}

#[test]
fn a_stream_that_asks_for_it_is_built_again_and_the_old_one_dropped_first() {
    let t = Tally::default();
    let tb = t.clone();
    let mut cap = supervised("t", RETRY, move |rebuild| {
        assert_eq!(
            tb.alive.load(Ordering::SeqCst),
            0,
            "the old stream is gone before the new one"
        );
        tb.built.fetch_add(1, Ordering::SeqCst);
        tb.alive.fetch_add(1, Ordering::SeqCst);
        tb.handles.lock().unwrap().push(rebuild.clone());
        Ok(Stream(tb.clone()))
    })
    .unwrap();

    t.handles.lock().unwrap()[0].request(); // as an error callback would
    assert!(until(|| t.built.load(Ordering::SeqCst) == 2));
    assert_eq!(t.alive.load(Ordering::SeqCst), 1);

    cap.stop();
    assert_eq!(
        t.alive.load(Ordering::SeqCst),
        0,
        "stopping drops the stream"
    );
}

#[test]
fn a_rebuild_that_fails_is_tried_again_until_it_works() {
    let t = Tally::default();
    let tb = t.clone();
    let mut cap = supervised("t", RETRY, move |rebuild| {
        let n = tb.built.fetch_add(1, Ordering::SeqCst);
        tb.handles.lock().unwrap().push(rebuild.clone());
        if (1..4).contains(&n) {
            return Err(CaptureError::NoDevice); // the new device is not there yet
        }
        tb.alive.fetch_add(1, Ordering::SeqCst);
        Ok(Stream(tb.clone()))
    })
    .unwrap();

    t.handles.lock().unwrap()[0].request();
    assert!(until(
        || t.built.load(Ordering::SeqCst) == 5 && t.alive.load(Ordering::SeqCst) == 1
    ));
    cap.stop();
    assert_eq!(t.alive.load(Ordering::SeqCst), 0);
}

#[test]
fn stop_is_prompt_while_waiting_to_retry() {
    let t = Tally::default();
    let tb = t.clone();
    let mut cap = supervised("t", Duration::from_secs(60), move |rebuild| {
        let n = tb.built.fetch_add(1, Ordering::SeqCst);
        tb.handles.lock().unwrap().push(rebuild.clone());
        if n > 0 {
            return Err(CaptureError::NoDevice);
        }
        tb.alive.fetch_add(1, Ordering::SeqCst);
        Ok(Stream(tb.clone()))
    })
    .unwrap();
    t.handles.lock().unwrap()[0].request();
    assert!(until(|| t.built.load(Ordering::SeqCst) == 2));

    let started = Instant::now();
    cap.stop();
    assert!(
        started.elapsed() < Duration::from_secs(1),
        "{:?}",
        started.elapsed()
    );
}

#[test]
fn a_changed_or_vanished_device_ends_the_stream_and_little_else_does() {
    use cpal::ErrorKind::*;
    use meetbutler_audio::mic::ends_stream;
    assert!(
        ends_stream(StreamInvalidated),
        "WASAPI reports a new default device this way"
    );
    assert!(
        ends_stream(DeviceNotAvailable),
        "unplugged with no replacement"
    );
    // rerouted already (the stream follows the default), or passing trouble
    for kind in [DeviceChanged, Xrun, DeviceBusy, RealtimeDenied] {
        assert!(!ends_stream(kind), "{kind:?}");
    }
}

#[test]
fn a_stream_found_stale_is_built_again_without_its_callback_asking() {
    // the default microphone changed (a headset plugged in): the old stream still runs,
    // on the old device, and says nothing. The test waits on how many times the stream
    // was asked, not on the clock, so a slow machine only makes it slower.
    use meetbutler_audio::supervise::supervised_watching;
    let t = Tally::default();
    let tb = t.clone();
    let switched = Arc::new(AtomicUsize::new(0)); // which device is the default now
    let asked = Arc::new(AtomicUsize::new(0)); // how many times `stale` ran
    let (sw_build, sw_check, asked_in) = (switched.clone(), switched.clone(), asked.clone());
    let mut cap = supervised_watching(
        "t",
        RETRY,
        RETRY,
        move |_| {
            tb.built.fetch_add(1, Ordering::SeqCst);
            tb.alive.fetch_add(1, Ordering::SeqCst);
            Ok((Stream(tb.clone()), sw_build.load(Ordering::SeqCst)))
        },
        move |(_, device): &(Stream, usize)| {
            asked_in.fetch_add(1, Ordering::SeqCst);
            *device != sw_check.load(Ordering::SeqCst)
        },
    )
    .unwrap();
    let asked_again = |times: usize| {
        let from = asked.load(Ordering::SeqCst);
        assert!(
            until(|| asked.load(Ordering::SeqCst) >= from + times),
            "never asked"
        );
    };

    asked_again(3);
    assert_eq!(
        t.built.load(Ordering::SeqCst),
        1,
        "nothing changed, nothing rebuilt"
    );
    switched.store(1, Ordering::SeqCst);
    assert!(
        until(|| t.built.load(Ordering::SeqCst) == 2),
        "built on the new default"
    );
    asked_again(3);
    assert_eq!(t.built.load(Ordering::SeqCst), 2, "and only once");
    cap.stop();
    assert_eq!(t.alive.load(Ordering::SeqCst), 0);
}
