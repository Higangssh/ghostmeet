//! The sidecar's request loop.
//!
//! A reader thread takes lines off stdin, so a `cancel` lands while a generation is
//! running; the loop itself is the only writer of stdout. When stdin closes (the app quit
//! or died) the running generation is stopped and the loop returns.

use std::collections::HashSet;
use std::io::{BufRead, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::Instant;

use crate::protocol::{ErrorKind, Event, GenerateRequest, Generated, Phase, Request, Stats};

/// A language model behind the loop. `on_progress(phase, done, total)` is called as the
/// prompt is read and as tokens come out, and returns false to stop.
pub trait Generator {
    fn load(&mut self, model: &Path) -> Result<(), String>;
    fn generate(
        &mut self,
        req: &GenerateRequest,
        on_progress: &mut dyn FnMut(Phase, u32, u32) -> bool,
    ) -> Result<Generated, String>;
}

/// How often progress goes out while generating, in tokens.
const PROGRESS_EVERY: u32 = 8;

enum Msg {
    Generate(GenerateRequest),
    Bad { id: Option<String>, message: String },
    Stop,
}

#[derive(Default)]
struct Cancels {
    running: Option<String>,
    flag: Arc<AtomicBool>,
    waiting: HashSet<String>,
}

/// Serve requests from `input` until it closes or a `shutdown` arrives.
pub fn serve<R, W, G>(input: R, mut out: W, mut gen: G)
where
    R: BufRead + Send + 'static,
    W: Write,
    G: Generator,
{
    let cancels = Arc::new(Mutex::new(Cancels::default()));
    let (tx, rx) = mpsc::channel();
    {
        let cancels = cancels.clone();
        std::thread::spawn(move || read_requests(input, tx, cancels));
    }

    let mut loaded: Option<PathBuf> = None;
    while let Ok(msg) = rx.recv() {
        match msg {
            Msg::Stop => break,
            Msg::Bad { id, message } => emit(
                &mut out,
                Event::Error {
                    id,
                    kind: ErrorKind::BadRequest,
                    message,
                },
            ),
            Msg::Generate(req) => {
                let flag = {
                    let mut c = cancels.lock().unwrap_or_else(|e| e.into_inner());
                    if c.waiting.remove(&req.id) {
                        drop(c);
                        cancelled(&mut out, &req.id);
                        continue;
                    }
                    c.running = Some(req.id.clone());
                    c.flag = Arc::new(AtomicBool::new(false));
                    c.flag.clone()
                };
                run(&mut out, &mut gen, &mut loaded, &req, &flag);
                cancels.lock().unwrap_or_else(|e| e.into_inner()).running = None;
            }
        }
    }
}

fn read_requests<R: BufRead>(input: R, tx: mpsc::Sender<Msg>, cancels: Arc<Mutex<Cancels>>) {
    for line in input.lines() {
        let Ok(line) = line else { break };
        if line.trim().is_empty() {
            continue;
        }
        let value: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(e) => {
                let _ = tx.send(Msg::Bad {
                    id: None,
                    message: format!("not JSON: {e}"),
                });
                continue;
            }
        };
        let id = value.get("id").and_then(|v| v.as_str()).map(str::to_string);
        match serde_json::from_value::<Request>(value) {
            Ok(Request::Generate(req)) => {
                let _ = tx.send(Msg::Generate(req));
            }
            Ok(Request::Cancel { id }) => {
                let mut c = cancels.lock().unwrap_or_else(|e| e.into_inner());
                if c.running.as_deref() == Some(id.as_str()) {
                    c.flag.store(true, Ordering::SeqCst);
                } else {
                    c.waiting.insert(id);
                }
            }
            Ok(Request::Shutdown) => {
                stop_running(&cancels);
                let _ = tx.send(Msg::Stop);
                return;
            }
            Err(e) => {
                let _ = tx.send(Msg::Bad {
                    id,
                    message: e.to_string(),
                });
            }
        }
    }
    // stdin closed: the app is gone, so is any reason to keep generating
    stop_running(&cancels);
    let _ = tx.send(Msg::Stop);
}

fn stop_running(cancels: &Mutex<Cancels>) {
    cancels
        .lock()
        .unwrap_or_else(|e| e.into_inner())
        .flag
        .store(true, Ordering::SeqCst);
}

fn run<W: Write, G: Generator>(
    out: &mut W,
    gen: &mut G,
    loaded: &mut Option<PathBuf>,
    req: &GenerateRequest,
    cancel: &AtomicBool,
) {
    let mut load_sec = 0.0;
    if loaded.as_deref() != Some(req.model.as_path()) {
        let started = Instant::now();
        if let Err(message) = gen.load(&req.model) {
            *loaded = None;
            emit(
                out,
                Event::Error {
                    id: Some(req.id.clone()),
                    kind: ErrorKind::Model,
                    message,
                },
            );
            return;
        }
        load_sec = started.elapsed().as_secs_f64();
        *loaded = Some(req.model.clone());
        emit(
            out,
            Event::Loaded {
                id: req.id.clone(),
                model: req.model.clone(),
                sec: load_sec,
            },
        );
    }

    let result = gen.generate(req, &mut |phase, done, total| {
        if cancel.load(Ordering::SeqCst) {
            return false;
        }
        // the prompt is reported once per batch; tokens are many, so only every few
        let report =
            phase == Phase::Prompt || done == 1 || done % PROGRESS_EVERY == 0 || done == total;
        if report {
            emit(
                &mut *out,
                Event::Progress {
                    id: req.id.clone(),
                    phase,
                    done,
                    total,
                },
            );
        }
        true
    });

    if cancel.load(Ordering::SeqCst) {
        cancelled(out, &req.id);
        return;
    }
    match result {
        Ok(g) => {
            let stats = Stats {
                load_sec,
                prompt_tokens: g.prompt_tokens,
                gen_tokens: g.gen_tokens,
                prompt_sec: g.prompt_sec,
                gen_sec: g.gen_sec,
            };
            emit(
                out,
                Event::Done {
                    id: req.id.clone(),
                    text: g.text,
                    stats,
                },
            );
        }
        Err(message) => emit(
            out,
            Event::Error {
                id: Some(req.id.clone()),
                kind: ErrorKind::Generate,
                message,
            },
        ),
    }
}

fn cancelled<W: Write>(out: &mut W, id: &str) {
    emit(
        out,
        Event::Error {
            id: Some(id.to_string()),
            kind: ErrorKind::Cancelled,
            message: "cancelled".into(),
        },
    );
}

fn emit<W: Write>(out: &mut W, event: Event) {
    // nothing to do if the app stopped reading; the reader thread will see stdin close
    if let Ok(line) = serde_json::to_string(&event) {
        let _ = writeln!(out, "{line}");
        let _ = out.flush();
    }
}
