//! The span rule, checked against the VAD the benchmark measured with.
//!
//! `fixtures/silero_synth.json` comes from `bench/silero_reference.py` on the synthetic
//! session from `bench/make_synth.sh`: the per-frame probabilities of the benchmark's
//! Silero model, and the spans sherpa-onnx cut from the same audio. Reproducing those
//! spans exactly from those probabilities is what lets the benchmark numbers stand for
//! this engine.

use meetbutler_engine::spans::{speech_spans, Span, SpanConfig, SpanCutter, BENCH};
use serde_json::Value;

fn fixture() -> Value {
    serde_json::from_str(include_str!("fixtures/silero_synth.json")).unwrap()
}

fn spans_of(v: &Value) -> Vec<Span> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| Span::new(p[0].as_u64().unwrap() as usize, p[1].as_u64().unwrap() as usize))
        .collect()
}

fn probs_of(v: &Value) -> Vec<f32> {
    v["probs"].as_array().unwrap().iter().map(|p| p.as_f64().unwrap() as f32).collect()
}

#[test]
fn the_bench_values_are_the_ones_the_numbers_were_measured_with() {
    let vad = &fixture()["vad"];
    assert_eq!(BENCH.threshold, vad["threshold"].as_f64().unwrap() as f32);
    assert_eq!(BENCH.min_silence_sec, vad["min_silence_sec"].as_f64().unwrap() as f32);
    assert_eq!(BENCH.min_speech_sec, vad["min_speech_sec"].as_f64().unwrap() as f32);
    assert_eq!(BENCH.max_speech_sec, vad["max_speech_sec"].as_f64().unwrap() as f32);
    assert_eq!(BENCH.pad_sec, vad["pad_sec"].as_f64().unwrap() as f32);
}

#[test]
fn unpadded_spans_match_sherpa_exactly() {
    let f = fixture();
    let mut cutter = SpanCutter::new(BENCH);
    let mut got: Vec<Span> = probs_of(&f).into_iter().filter_map(|p| cutter.push(p)).collect();
    got.extend(cutter.finish());

    assert_eq!(got, spans_of(&f["spans_unpadded"]));
}

#[test]
fn padded_spans_match_the_benchmark_exactly() {
    let f = fixture();
    let total = f["total_samples"].as_u64().unwrap() as usize;

    assert_eq!(speech_spans(&probs_of(&f), total, &BENCH), spans_of(&f["spans"]));
}

// --- the rule itself, on hand-made probabilities ------------------------------------

const FRAME: usize = 512;

/// `secs` worth of frames at probability `p`.
fn run(p: f32, secs: f32) -> Vec<f32> {
    vec![p; (secs * 16_000.0 / FRAME as f32).round() as usize]
}

fn cat(parts: &[Vec<f32>]) -> Vec<f32> {
    parts.concat()
}

fn secs(spans: &[Span]) -> Vec<(f32, f32)> {
    spans.iter().map(|s| (s.start_sec(), s.end_sec())).collect()
}

#[test]
fn silence_has_no_spans() {
    let probs = run(0.05, 10.0);
    assert!(speech_spans(&probs, probs.len() * FRAME, &BENCH).is_empty());
}

#[test]
fn a_blip_shorter_than_min_speech_is_dropped() {
    let probs = cat(&[run(0.0, 2.0), run(0.9, 0.15), run(0.0, 2.0)]);
    assert!(speech_spans(&probs, probs.len() * FRAME, &BENCH).is_empty());
}

#[test]
fn a_short_reply_survives() {
    // "네" — about half a second, well over min_speech
    let probs = cat(&[run(0.0, 2.0), run(0.9, 0.5), run(0.0, 2.0)]);
    let spans = speech_spans(&probs, probs.len() * FRAME, &BENCH);
    assert_eq!(spans.len(), 1);
    let s = spans[0];
    assert!(s.start_sec() < 2.0 && s.end_sec() > 2.5, "{:?}", secs(&spans));
}

#[test]
fn a_pause_shorter_than_min_silence_does_not_split() {
    let probs = cat(&[run(0.0, 1.0), run(0.9, 3.0), run(0.0, 0.3), run(0.9, 3.0), run(0.0, 2.0)]);
    assert_eq!(speech_spans(&probs, probs.len() * FRAME, &BENCH).len(), 1);
}

#[test]
fn a_pause_longer_than_min_silence_splits() {
    let probs = cat(&[run(0.0, 1.0), run(0.9, 3.0), run(0.0, 1.0), run(0.9, 3.0), run(0.0, 2.0)]);
    assert_eq!(speech_spans(&probs, probs.len() * FRAME, &BENCH).len(), 2);
}

#[test]
fn speech_running_into_the_end_is_kept() {
    let probs = cat(&[run(0.0, 1.0), run(0.9, 3.0)]);
    let total = probs.len() * FRAME;
    let spans = speech_spans(&probs, total, &BENCH);
    assert_eq!(spans.len(), 1);
    assert_eq!(spans[0].end, total, "padding is clipped to the audio");
}

#[test]
fn padding_is_clipped_at_the_start() {
    let probs = cat(&[run(0.9, 3.0), run(0.0, 2.0)]);
    assert_eq!(speech_spans(&probs, probs.len() * FRAME, &BENCH)[0].start, 0);
}

#[test]
fn past_max_speech_a_brief_dip_is_enough_to_split() {
    // over max_speech_sec the rule tightens (threshold 0.9, min silence 0.1 s), so a
    // short dip that would not split normal speech ends the span
    let dip = run(0.5, 0.2);
    let probs = cat(&[run(0.0, 1.0), run(0.95, 26.0), dip.clone(), run(0.95, 5.0), run(0.0, 2.0)]);
    assert_eq!(speech_spans(&probs, probs.len() * FRAME, &BENCH).len(), 2);

    let probs = cat(&[run(0.0, 1.0), run(0.95, 10.0), dip, run(0.95, 5.0), run(0.0, 2.0)]);
    assert_eq!(speech_spans(&probs, probs.len() * FRAME, &BENCH).len(), 1);
}

#[test]
fn no_span_is_longer_than_a_whisper_window() {
    // sherpa does not enforce max_speech when speech never dips (the synthetic session
    // has a 29 s span); Whisper decodes at most 30 s at once, so longer spans are split
    let probs = cat(&[run(0.0, 1.0), run(0.99, 75.0), run(0.0, 2.0)]);
    let spans = speech_spans(&probs, probs.len() * FRAME, &BENCH);
    assert!(spans.len() >= 3, "{:?}", secs(&spans));
    assert!(spans.iter().all(|s| s.len() <= 30 * 16_000), "{:?}", secs(&spans));
    for w in spans.windows(2) {
        assert!(w[0].end <= w[1].start + 2 * BENCH.pad_samples(), "{:?}", secs(&spans));
    }
}

#[test]
fn a_custom_config_is_honoured() {
    let strict = SpanConfig { threshold: 0.95, ..BENCH };
    let probs = cat(&[run(0.0, 1.0), run(0.9, 3.0), run(0.0, 2.0)]);
    assert!(speech_spans(&probs, probs.len() * FRAME, &strict).is_empty());
}
