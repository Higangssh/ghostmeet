//! Slow tier: the engine's Silero against the VAD the benchmark measured with, on real
//! read speech (FLEURS) as well as synthetic speech. A VAD change judged on synthetic
//! speech alone once looked fine and lost 18% of the speech in FLEURS.
//!
//!     bench/make_synth.sh
//!     bench/.venv/bin/python -I bench/silero_reference.py bench/data/fleurs_ref/session.wav \
//!         --fleurs 10 --out bench/data/fleurs_ref/session.silero.json
//!     cargo test -p meetbutler-engine --features silero --test slow_vad -- --ignored --nocapture
//!
//! Reads `bench/data/models/silero_vad.onnx` (or `MEETBUTLER_MODELS`); no network.
#![cfg(feature = "silero")]

use std::path::{Path, PathBuf};

use meetbutler_engine::silero::SileroOnnx;
use meetbutler_engine::span_decoder::SpeechProbs;
use meetbutler_engine::spans::{speech_spans, Span, BENCH};
use serde_json::Value;

fn repo() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../..")
}

fn vad() -> SileroOnnx {
    let dir = std::env::var_os("MEETBUTLER_MODELS")
        .map_or_else(|| repo().join("bench/data/models"), PathBuf::from);
    SileroOnnx::load(dir.join("silero_vad.onnx")).unwrap()
}

fn read_wav(path: &Path) -> Vec<f32> {
    let mut r = hound::WavReader::open(path).unwrap_or_else(|e| panic!("{}: {e}", path.display()));
    r.samples::<i16>()
        .map(|s| s.unwrap() as f32 / 32768.0)
        .collect()
}

fn reference(path: &Path) -> Value {
    let text = std::fs::read_to_string(path)
        .unwrap_or_else(|e| panic!("{}: {e} — see the header", path.display()));
    serde_json::from_str(&text).unwrap()
}

fn spans(v: &Value) -> Vec<Span> {
    v.as_array()
        .unwrap()
        .iter()
        .map(|p| {
            Span::new(
                p[0].as_u64().unwrap() as usize,
                p[1].as_u64().unwrap() as usize,
            )
        })
        .collect()
}

fn probs(v: &Value) -> Vec<f32> {
    v["probs"]
        .as_array()
        .unwrap()
        .iter()
        .map(|p| p.as_f64().unwrap() as f32)
        .collect()
}

/// Worst frame difference between the engine's probabilities and onnxruntime's.
fn worst_prob_error(wav: &Path, reference: &Value) -> f32 {
    let audio = read_wav(wav);
    assert_eq!(
        audio.len() as u64,
        reference["total_samples"].as_u64().unwrap(),
        "audio and reference differ"
    );
    let got = vad().probs(&audio).unwrap();
    let want = probs(reference);
    assert_eq!(got.len(), want.len());
    got.iter()
        .zip(&want)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0, f32::max)
}

#[test]
#[ignore = "needs bench/data models and audio"]
fn probabilities_match_the_benchmark_on_synthetic_speech() {
    let worst = worst_prob_error(
        &repo().join("bench/data/synth/session.wav"),
        &serde_json::from_str(include_str!("fixtures/silero_synth.json")).unwrap(),
    );
    eprintln!("synthetic: worst frame difference {worst:.2e}");
    assert!(worst <= 1e-4, "{worst}");
}

#[test]
#[ignore = "needs bench/data models and audio"]
fn probabilities_match_the_benchmark_on_fleurs() {
    let dir = repo().join("bench/data/fleurs_ref");
    let worst = worst_prob_error(
        &dir.join("session.wav"),
        &reference(&dir.join("session.silero.json")),
    );
    eprintln!("FLEURS: worst frame difference {worst:.2e}");
    assert!(worst <= 1e-4, "{worst}");
}

#[test]
#[ignore = "needs bench/data audio"]
fn the_span_rule_matches_sherpa_on_fleurs() {
    let r = reference(&repo().join("bench/data/fleurs_ref/session.silero.json"));
    let total = r["total_samples"].as_u64().unwrap() as usize;
    let got = speech_spans(&probs(&r), total, &BENCH);
    let want = spans(&r["spans"]);
    let first = got.iter().zip(&want).position(|(a, b)| a != b);
    assert_eq!(got.len(), want.len(), "first difference at {first:?}");
    assert_eq!(
        first,
        None,
        "first difference: {:?} vs {:?}",
        first.map(|i| got[i]),
        first.map(|i| want[i])
    );
}
