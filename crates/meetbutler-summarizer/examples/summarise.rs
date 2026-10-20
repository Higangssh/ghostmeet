//! The app's summary over a transcript file, for the benchmark: what the app would keep.
//!
//!     cargo run --release -p meetbutler-summarizer --features llama --example summarise -- \
//!         SIDECAR MODEL.gguf TRANSCRIPT.json [--record ANSWERS.json | --replay ANSWERS.json]
//!
//! TRANSCRIPT.json is `{"lines": [...]}` (or `"transcript"`), one line per segment, as
//! `bench/meeting_gen.py` and `bench/asr_meetings.py` write them. `--record` keeps every
//! answer the model gave; `--replay` answers from such a file instead of the model, so
//! the merge step can be changed and scored again without running the model: decoding
//! is greedy, so the same prompts get the same answers.

use std::sync::atomic::AtomicBool;

use meetbutler_summarizer::client::{ClientError, Sidecar};
use meetbutler_summarizer::protocol::{Event, GenerateRequest, Stats};
use meetbutler_summarizer::summary::{summarise, Generate, Options};

struct Record<'a> {
    inner: &'a mut Sidecar,
    answers: Vec<String>,
}

impl Generate for Record<'_> {
    fn generate(
        &mut self,
        req: GenerateRequest,
        on_event: &mut dyn FnMut(&Event) -> bool,
    ) -> Result<(String, Stats), ClientError> {
        let out = Generate::generate(self.inner, req, on_event)?;
        self.answers.push(out.0.clone());
        Ok(out)
    }
}

struct Replay(std::vec::IntoIter<String>);

impl Generate for Replay {
    fn generate(
        &mut self,
        _: GenerateRequest,
        _: &mut dyn FnMut(&Event) -> bool,
    ) -> Result<(String, Stats), ClientError> {
        let text = self.0.next().ok_or(ClientError::Exited(None))?;
        Ok((text, Stats::default()))
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let (sidecar, model, transcript, mode) = match &args[..] {
        [s, m, t] => (s, m, t, None),
        [s, m, t, flag, file] if flag == "--record" || flag == "--replay" => {
            (s, m, t, Some((flag.as_str(), file)))
        }
        _ => {
            return Err(
                "usage: summarise SIDECAR MODEL.gguf TRANSCRIPT.json [--record|--replay FILE]"
                    .into(),
            )
        }
    };
    let meeting: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(transcript)?)?;
    let lines: Vec<String> = serde_json::from_value(
        meeting
            .get("transcript")
            .or_else(|| meeting.get("lines"))
            .cloned()
            .ok_or("no lines")?,
    )?;
    let opts = Options {
        model: model.into(),
    };
    let progress = |p: f64| eprintln!("{:.0}%", p * 100.0);
    let cancel = AtomicBool::new(false);
    let s = match mode {
        Some(("--replay", file)) => {
            let answers: Vec<String> = serde_json::from_str(&std::fs::read_to_string(file)?)?;
            summarise(
                &mut Replay(answers.into_iter()),
                &lines,
                &opts,
                &cancel,
                progress,
            )?
        }
        Some((_, file)) => {
            let mut sidecar = Sidecar::spawn(sidecar.as_ref(), &[])?;
            let mut rec = Record {
                inner: &mut sidecar,
                answers: Vec::new(),
            };
            let s = summarise(&mut rec, &lines, &opts, &cancel, progress)?;
            std::fs::write(file, serde_json::to_string(&rec.answers)?)?;
            s
        }
        None => {
            let mut sidecar = Sidecar::spawn(sidecar.as_ref(), &[])?;
            summarise(&mut sidecar, &lines, &opts, &cancel, progress)?
        }
    };
    println!("{}", serde_json::to_string(&s)?);
    Ok(())
}
