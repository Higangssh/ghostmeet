//! The gap filler: a source that stops delivering while nothing plays (WASAPI loopback,
//! perhaps ScreenCaptureKit) must still advance with the clock, or every segment after a
//! quiet stretch is stamped early and "me" and "them" stop lining up.

use meetbutler_audio::gap::{GapFiller, FILL_AFTER_SEC};
use meetbutler_audio::TARGET_RATE;

const RATE: f64 = TARGET_RATE as f64;

/// `n` samples of a constant, standing for real audio.
fn audio(n: usize) -> Vec<f32> {
    vec![0.5; n]
}

/// Feed packets of `len` samples, each stamped at `start + i * len / RATE * speed`.
fn feed(g: &mut GapFiller, start: f64, packets: usize, len: usize, speed: f64, out: &mut Vec<f32>) {
    for i in 0..packets {
        g.push(
            start + i as f64 * len as f64 / RATE * speed,
            &audio(len),
            out,
        );
    }
}

fn zeros(out: &[f32]) -> usize {
    out.iter().filter(|&&s| s == 0.0).count()
}

#[test]
fn audio_that_keeps_coming_passes_through_unchanged() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    feed(&mut g, 100.0, 500, 160, 1.0, &mut out);
    assert_eq!(out.len(), 500 * 160);
    assert_eq!(zeros(&out), 0);
}

#[test]
fn a_quiet_stretch_with_no_data_becomes_that_much_silence() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    feed(&mut g, 0.0, 100, 160, 1.0, &mut out); // one second
                                                // nothing for three seconds, then audio again
    feed(&mut g, 4.0, 100, 160, 1.0, &mut out);

    assert_eq!(zeros(&out), 3 * TARGET_RATE as usize);
    assert_eq!(
        out.len(),
        5 * TARGET_RATE as usize,
        "the channel is as long as the clock says"
    );
    // the silence sits between the two runs of audio
    assert!(out[..16_000].iter().all(|&s| s == 0.5));
    assert!(out[16_000..64_000].iter().all(|&s| s == 0.0));
    assert!(out[64_000..].iter().all(|&s| s == 0.5));
}

#[test]
fn jitter_under_the_threshold_adds_nothing() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    let late = FILL_AFTER_SEC * 0.8;
    for i in 0..300 {
        let t = i as f64 * 0.01 + if i % 7 == 3 { late } else { 0.0 };
        g.push(t, &audio(160), &mut out);
    }
    assert_eq!(zeros(&out), 0);
    assert_eq!(out.len(), 300 * 160);
}

#[test]
fn a_packet_stamped_early_is_kept_whole_never_trimmed() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    feed(&mut g, 0.0, 10, 160, 1.0, &mut out);
    g.push(0.0, &audio(160), &mut out); // stamped a tenth of a second early
    assert_eq!(out.len(), 11 * 160);
    assert_eq!(zeros(&out), 0);
}

#[test]
fn four_hours_of_steady_packets_never_drift_into_filling() {
    // the long-session requirement: 4 h and more
    let mut g = GapFiller::new();
    let (mut out, packet) = (Vec::new(), audio(16_000));
    let mut filled = 0;
    for i in 0..4 * 3600u64 {
        out.clear();
        g.push(i as f64, &packet, &mut out);
        filled += out.len() - packet.len();
    }
    assert_eq!(filled, 0);
}

#[test]
fn a_device_clock_running_fast_is_absorbed_by_the_next_gap() {
    // the device delivers 0.01 % more samples than the clock says: after an hour the
    // channel is 0.36 s ahead, so a 5 s silence is filled with 5 - 0.36 s
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    let speed = 1.0 / 1.0001;
    feed(&mut g, 0.0, 36_000, 1600, speed, &mut out); // one hour of samples
    let hour = out.len();
    let resume = 36_000.0 * 0.1 * speed + 5.0;
    out.clear();
    g.push(resume, &audio(160), &mut out);

    let filled = zeros(&out) as f64 / RATE;
    let ahead = hour as f64 / RATE - 36_000.0 * 0.1 * speed;
    assert!(
        (filled - (5.0 - ahead)).abs() < 0.01,
        "filled {filled:.3} s, ahead {ahead:.3} s"
    );
}

#[test]
fn an_empty_packet_still_moves_the_clock() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    g.push(0.0, &audio(160), &mut out);
    g.push(2.0, &[], &mut out);
    assert_eq!(out.len(), 160 + (2.0 * RATE) as usize - 160);
}

#[test]
fn the_silence_filled_is_counted() {
    let mut g = GapFiller::new();
    let mut out = Vec::new();
    g.push(0.0, &audio(160), &mut out);
    g.push(3.01, &audio(160), &mut out);
    g.push(3.02, &audio(160), &mut out);
    assert!((g.filled_sec() - 3.0).abs() < 1e-3, "{}", g.filled_sec());
}
