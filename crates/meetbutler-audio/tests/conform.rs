//! Conform — any device format in, 16 kHz mono out, chunk by chunk.
//!
//! Devices deliver 44.1 or 48 kHz, often stereo; the engine wants 16 kHz mono. The
//! conversion runs inside capture callbacks on arbitrary chunk sizes, so it has to give
//! the same result however the audio is cut, and must not let content above 8 kHz fold
//! back into the speech band.

use std::f64::consts::PI;

use meetbutler_audio::conform::{Conformer, Resampler};

fn sine(freq: f64, rate: u32, secs: f64, amp: f64) -> Vec<f32> {
    (0..(secs * rate as f64) as usize)
        .map(|i| (amp * (2.0 * PI * freq * i as f64 / rate as f64).sin()) as f32)
        .collect()
}

fn rms(x: &[f32]) -> f64 {
    (x.iter().map(|&v| (v as f64).powi(2)).sum::<f64>() / x.len().max(1) as f64).sqrt()
}

fn run(r: &mut Resampler, input: &[f32], chunk: usize) -> Vec<f32> {
    let mut out = Vec::new();
    for c in input.chunks(chunk.max(1)) {
        r.process(c, &mut out);
    }
    r.flush(&mut out);
    out
}

#[test]
fn sixteen_kilohertz_passes_through_untouched() {
    let input = sine(440.0, 16_000, 0.5, 0.5);
    let out = run(&mut Resampler::new(16_000, 16_000), &input, 160);
    assert_eq!(out, input);
}

#[test]
fn the_output_length_follows_the_rate_ratio() {
    for (rate, secs) in [(48_000u32, 2.0), (44_100, 2.0), (32_000, 1.5), (8_000, 1.0)] {
        let input = vec![0.0f32; (secs * rate as f64) as usize];
        let out = run(&mut Resampler::new(rate, 16_000), &input, 480);
        let want = (input.len() as f64 * 16_000.0 / rate as f64).ceil() as usize;
        assert_eq!(out.len(), want, "{rate} Hz");
    }
}

#[test]
fn a_tone_in_the_speech_band_survives_downsampling() {
    for rate in [48_000u32, 44_100] {
        let input = sine(1000.0, rate, 1.0, 0.5);
        let out = run(&mut Resampler::new(rate, 16_000), &input, 512);
        let want = sine(1000.0, 16_000, 1.0, 0.5);
        // skip the edges, where the filter sees zeros beyond the signal
        let err = out[200..15_800]
            .iter()
            .zip(&want[200..15_800])
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(err < 0.01, "{rate} Hz: max error {err}");
    }
}

#[test]
fn content_above_eight_kilohertz_does_not_alias_into_speech() {
    // a 12 kHz tone at 48 kHz would fold back to 4 kHz without a low-pass
    let input = sine(12_000.0, 48_000, 1.0, 0.5);
    let out = run(&mut Resampler::new(48_000, 16_000), &input, 480);
    assert!(rms(&out[200..out.len() - 200]) < 0.01, "rms {}", rms(&out));
}

#[test]
fn chunk_boundaries_do_not_change_the_result() {
    let input = sine(700.0, 44_100, 1.0, 0.3);
    let whole = run(&mut Resampler::new(44_100, 16_000), &input, input.len());
    for chunk in [1, 7, 441, 1000, 4096] {
        let cut = run(&mut Resampler::new(44_100, 16_000), &input, chunk);
        assert_eq!(cut.len(), whole.len(), "chunk {chunk}");
        let diff = cut
            .iter()
            .zip(&whole)
            .map(|(a, b)| (a - b).abs())
            .fold(0.0f32, f32::max);
        assert!(diff < 1e-6, "chunk {chunk}: {diff}");
    }
}

#[test]
fn long_recordings_do_not_drift() {
    // output position is derived from a sample count, never accumulated; ten minutes is
    // millions of outputs, enough for accumulated rounding to show
    let mut r = Resampler::new(44_100, 16_000);
    let chunk = vec![0.0f32; 44_100];
    let mut out = Vec::new();
    let mut produced = 0usize;
    for _ in 0..600 {
        out.clear();
        r.process(&chunk, &mut out);
        produced += out.len();
    }
    out.clear();
    r.flush(&mut out);
    produced += out.len();
    assert_eq!(produced, 600 * 16_000);
}

#[test]
fn stereo_is_averaged_to_mono() {
    let mut c = Conformer::new(16_000, 2);
    let mut out = Vec::new();
    c.process(&[0.5, 0.5, 0.2, -0.2, 1.0, 0.0], &mut out);
    assert_eq!(out, [0.5, 0.0, 0.5]);
}

#[test]
fn a_trailing_partial_frame_waits_for_the_rest() {
    // a callback can end between the left and right sample of a frame
    let mut c = Conformer::new(16_000, 2);
    let mut out = Vec::new();
    c.process(&[0.4, 0.4, 0.6], &mut out);
    c.process(&[0.2], &mut out);
    assert_eq!(out, [0.4, 0.4]);
}

#[test]
fn a_conformer_downmixes_then_resamples() {
    let left = sine(1000.0, 48_000, 1.0, 0.5);
    let interleaved: Vec<f32> = left.iter().flat_map(|&s| [s, s]).collect();
    let mut c = Conformer::new(48_000, 2);
    let mut out = Vec::new();
    for chunk in interleaved.chunks(960) {
        c.process(chunk, &mut out);
    }
    c.flush(&mut out);
    assert_eq!(out.len(), 16_000);
    assert!((rms(&out[200..15_800]) - 0.5 / 2f64.sqrt()).abs() < 0.01);
}
