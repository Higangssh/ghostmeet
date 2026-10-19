//! PcmStore — append-only 16 kHz mono PCM backed by disk, with bounded-memory reads.
//!
//! This is the component that makes 4+ hour sessions possible: audio accumulates on
//! disk, and transcription only ever pulls a small tail window into RAM.

use meetbutler_engine::pcm_store::PcmStore;
use meetbutler_engine::{AudioSource, SAMPLE_RATE};

fn s16le(samples: &[i16]) -> Vec<u8> {
    samples.iter().flat_map(|s| s.to_le_bytes()).collect()
}

fn ramp(n: usize) -> Vec<i16> {
    (0..n).map(|i| i as i16).collect()
}

fn store(dir: &tempfile::TempDir) -> PcmStore {
    PcmStore::open(dir.path().join("audio.pcm")).unwrap()
}

fn close(a: f32, b: f32) -> bool {
    (a - b).abs() < 1e-6
}

#[test]
fn empty_store_has_no_samples() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    assert_eq!(s.total_samples(), 0);
    assert_eq!(s.duration(), 0.0);
}

#[test]
fn append_then_read_returns_normalized_float32() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&[0, 16384, -16384, 32767])).unwrap();

    let audio = s.read(0, None).unwrap();

    let want = [0.0, 0.5, -0.5, 32767.0 / 32768.0];
    assert_eq!(audio.len(), 4);
    assert!(
        audio.iter().zip(want).all(|(&a, b)| close(a, b)),
        "{audio:?}"
    );
}

#[test]
fn float_samples_are_stored_as_s16() {
    // capture hands over f32; the file stays s16le so an hour is ~115 MB, not ~230 MB
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append_f32(&[0.0, 0.5, -0.5, 2.0, -2.0]).unwrap();

    let audio = s.read(0, None).unwrap();

    assert_eq!(s.total_samples(), 5);
    let want = [0.0, 0.5, -0.5, 32767.0 / 32768.0, -1.0];
    assert!(
        audio.iter().zip(want).all(|(&a, b)| close(a, b)),
        "clipped, not wrapped: {audio:?}"
    );
}

#[test]
fn total_samples_and_duration_track_appends() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&ramp(SAMPLE_RATE))).unwrap(); // exactly 1 second
    assert_eq!(s.total_samples(), SAMPLE_RATE);
    assert!((s.duration() - 1.0).abs() < 1e-9);

    s.append(&s16le(&ramp(SAMPLE_RATE / 2))).unwrap(); // + half a second
    assert_eq!(s.total_samples(), SAMPLE_RATE + SAMPLE_RATE / 2);
    assert!((s.duration() - 1.5).abs() < 1e-9);
}

#[test]
fn read_returns_only_the_requested_window() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&ramp(1000))).unwrap();

    let audio = s.read(100, Some(150)).unwrap();

    assert_eq!(audio.len(), 50);
    assert!(close(audio[0], 100.0 / 32768.0));
    assert!(close(audio[49], 149.0 / 32768.0));
}

#[test]
fn read_past_end_clamps_instead_of_failing() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&ramp(10))).unwrap();

    assert_eq!(s.read(5, Some(9999)).unwrap().len(), 5);
}

#[test]
fn read_beyond_total_returns_empty() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&ramp(10))).unwrap();

    assert!(s.read(10, None).unwrap().is_empty());
    assert!(s.read(50, Some(60)).unwrap().is_empty());
}

#[test]
fn odd_trailing_byte_is_not_exposed_as_a_sample() {
    // a chunk can split an int16 across two appends — never emit half a sample
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    let mut five = s16le(&[1, 2]);
    five.push(3);
    s.append(&five).unwrap();

    assert_eq!(s.total_samples(), 2);

    s.append(&[0]).unwrap(); // completes the third sample

    assert_eq!(s.total_samples(), 3);
    assert!(close(s.read(2, Some(3)).unwrap()[0], 3.0 / 32768.0));
}

#[test]
fn reading_a_window_of_a_long_session_stays_bounded() {
    // a 4-hour session must not pull 4 hours of audio into RAM to transcribe its tail
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    let one_minute = vec![0u8; SAMPLE_RATE * 60 * 2];
    for _ in 0..60 {
        s.append(&one_minute).unwrap(); // 1 hour of audio on disk
    }
    assert!((s.duration() - 3600.0).abs() < 1e-6);

    let window = s.read(s.total_samples() - SAMPLE_RATE * 30, None).unwrap();

    assert_eq!(window.len(), SAMPLE_RATE * 30); // the window only, not the hour
}

#[test]
fn appending_after_close_is_dropped_not_raised() {
    // the capture thread can still be draining when a session tears down
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.append(&s16le(&[1, 2])).unwrap();
    s.close();

    s.append(&s16le(&[3, 4])).unwrap();

    assert_eq!(s.total_samples(), 2);
}

#[test]
fn close_is_idempotent() {
    let dir = tempfile::tempdir().unwrap();
    let s = store(&dir);
    s.close();
    s.close();
}

#[test]
fn data_survives_reopening_the_same_path() {
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("audio.pcm");
    let first = PcmStore::open(&path).unwrap();
    first.append(&s16le(&[1, 2, 3])).unwrap();
    first.close();

    let reopened = PcmStore::open(&path).unwrap();
    assert_eq!(reopened.total_samples(), 3);
    assert!(close(reopened.read(0, Some(1)).unwrap()[0], 1.0 / 32768.0));
}

#[test]
fn the_parent_directory_is_created() {
    let dir = tempfile::tempdir().unwrap();
    let s = PcmStore::open(dir.path().join("a/b/audio.pcm")).unwrap();
    s.append(&s16le(&[1])).unwrap();
    assert_eq!(s.total_samples(), 1);
}

#[test]
fn reads_and_appends_can_come_from_different_threads() {
    // capture appends while the transcription worker reads the tail
    let dir = tempfile::tempdir().unwrap();
    let s = std::sync::Arc::new(store(&dir));
    let writer = {
        let s = s.clone();
        std::thread::spawn(move || {
            for _ in 0..100 {
                s.append(&s16le(&ramp(160))).unwrap();
            }
        })
    };
    for _ in 0..100 {
        let n = s.total_samples();
        assert_eq!(s.read(0, Some(n)).unwrap().len(), n);
    }
    writer.join().unwrap();
    assert_eq!(s.total_samples(), 16_000);
}
