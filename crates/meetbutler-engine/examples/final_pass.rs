//! The after-meeting pass over one file, for the benchmark (`bench/engines.py`, `rust-*`).
//!
//!     cargo run --release --features whisper,silero --example final_pass -- \
//!         MODEL.bin silero_vad.onnx LANG AUDIO.pcm [VOCABULARY.txt] > segments.json
//!
//! AUDIO.pcm is raw 16 kHz mono s16le, the format a session channel is recorded in.
//! VOCABULARY.txt, one term per line, is the user's vocabulary (STT-7). Prints
//! `{"load_sec", "decode_sec", "segments": [[start, end, text], ...]}`.

use std::sync::Arc;
use std::time::Instant;

use meetbutler_engine::pcm_store::PcmStore;
use meetbutler_engine::silero::SileroOnnx;
use meetbutler_engine::span_decoder::final_transcript;
use meetbutler_engine::spans::BENCH;
use meetbutler_engine::whisper::{WhisperDecoder, WhisperModel};

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (model, vad, lang, audio, vocabulary) = match &args[..] {
        [m, v, l, a] => (m, v, l, a, Vec::new()),
        [m, v, l, a, words] => (
            m,
            v,
            l,
            a,
            std::fs::read_to_string(words)?
                .lines()
                .map(String::from)
                .collect(),
        ),
        _ => {
            return Err(
                "usage: final_pass MODEL.bin silero_vad.onnx LANG AUDIO.pcm [VOCABULARY.txt]"
                    .into(),
            )
        }
    };

    let t = Instant::now();
    let model = Arc::new(WhisperModel::load(model, true)?);
    let mut vad = SileroOnnx::load(vad)?;
    let load_sec = t.elapsed().as_secs_f64();

    let source = PcmStore::open(audio)?;
    source.close();
    let t = Instant::now();
    let mut decoder =
        WhisperDecoder::new(model.clone(), Some(lang.clone()))?.with_vocabulary(&vocabulary);
    let segments = final_transcript(&source, &mut vad, &mut decoder, &BENCH, 600.0, |_| {})?;
    let decode_sec = t.elapsed().as_secs_f64();

    let rows: Vec<serde_json::Value> = segments
        .iter()
        .map(|s| serde_json::json!([s.start, s.end, s.text]))
        .collect();
    println!(
        "{}",
        serde_json::json!({ "load_sec": load_sec, "decode_sec": decode_sec, "segments": rows })
    );
    drop(decoder);
    drop(model); // before exit: see WhisperModel
    Ok(())
}
