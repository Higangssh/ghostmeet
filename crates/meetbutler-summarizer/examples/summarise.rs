//! The app's summary over a transcript file, for the benchmark: what the app would keep.
//!
//!     cargo run --release -p meetbutler-summarizer --features llama --example summarise -- \
//!         SIDECAR MODEL.gguf TRANSCRIPT.json > summary.json
//!
//! TRANSCRIPT.json is `{"lines": [...]}` (or `"transcript"`), one line per segment, as
//! `bench/meeting_gen.py` and `bench/asr_meetings.py` write them.

use std::sync::atomic::AtomicBool;

use meetbutler_summarizer::client::Sidecar;
use meetbutler_summarizer::summary::{summarise, Options};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let [sidecar, model, transcript] = &args[..] else {
        return Err("usage: summarise SIDECAR MODEL.gguf TRANSCRIPT.json".into());
    };
    let meeting: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(transcript)?)?;
    let lines: Vec<String> = serde_json::from_value(
        meeting
            .get("transcript")
            .or_else(|| meeting.get("lines"))
            .cloned()
            .ok_or("no lines")?,
    )?;
    let mut sidecar = Sidecar::spawn(sidecar.as_ref(), &[])?;
    let opts = Options {
        model: model.into(),
    };
    let s = summarise(&mut sidecar, &lines, &opts, &AtomicBool::new(false), |p| {
        eprintln!("{:.0}%", p * 100.0)
    })?;
    println!("{}", serde_json::to_string(&s)?);
    Ok(())
}
