//! Speech spans of one file by the engine's rule, for the benchmark.
//!
//!     cargo run --release --features silero --example vad_spans -- silero_vad.onnx AUDIO.pcm
//!
//! AUDIO.pcm is raw 16 kHz mono s16le. Prints `[[start_sample, end_sample], ...]`, cut the
//! way the final pass cuts: Silero over 600 s chunks (`CHUNK_SEC`), one state machine.

use meetbutler_engine::pcm_store::PcmStore;
use meetbutler_engine::silero::SileroOnnx;
use meetbutler_engine::span_decoder::SpeechProbs;
use meetbutler_engine::spans::{finish_span, SpanCutter, BENCH, FRAME};
use meetbutler_engine::AudioSource;

fn main() -> Result<(), Box<dyn std::error::Error + Send + Sync>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [vad, audio] = &args[..] else {
        return Err("usage: vad_spans silero_vad.onnx AUDIO.pcm".into());
    };
    let mut vad = SileroOnnx::load(vad)?;
    let source = PcmStore::open(audio)?;
    source.close();
    let total = source.total_samples();
    let chunk_sec: usize = std::env::var("CHUNK_SEC")
        .ok()
        .and_then(|v| v.parse().ok())
        .unwrap_or(600);
    let chunk = chunk_sec * 16_000 / FRAME * FRAME;
    let mut cutter = SpanCutter::new(BENCH);
    let mut spans = Vec::new();
    let mut pos = 0;
    while pos < total {
        let end = (pos + chunk).min(total);
        let audio = source.read(pos, Some(end))?;
        let probs = vad.probs(&audio)?;
        for &p in &probs[..audio.len() / FRAME] {
            spans.extend(cutter.push(p));
        }
        pos = end;
    }
    spans.extend(cutter.finish());
    let out: Vec<[usize; 2]> = spans
        .into_iter()
        .flat_map(|s| finish_span(s, total, &BENCH))
        .map(|s| [s.start, s.end])
        .collect();
    println!("{}", serde_json::to_string(&out)?);
    Ok(())
}
