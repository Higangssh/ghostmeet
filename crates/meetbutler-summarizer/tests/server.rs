//! The sidecar's request loop, in process, with fake generators: the protocol, progress,
//! cancelling, bad input, and ending when the app goes away.

use std::io::{BufReader, Read, Write};
use std::path::Path;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{mpsc, Arc, Mutex};
use std::time::{Duration, Instant};

use meetbutler_summarizer::protocol::{Event, Generated, Phase};
use meetbutler_summarizer::server::{serve, Generator};
use serde_json::{json, Value};

/// Says `text` one character per token, `delay` apart.
#[derive(Clone)]
struct Fake {
    text: &'static str,
    delay: Duration,
    loads: Arc<AtomicUsize>,
    contexts: Arc<Mutex<Vec<u32>>>,
}

impl Fake {
    fn new(text: &'static str) -> Self {
        Self {
            text,
            delay: Duration::ZERO,
            loads: Arc::default(),
            contexts: Arc::default(),
        }
    }
    fn slow(text: &'static str, ms: u64) -> Self {
        Self {
            delay: Duration::from_millis(ms),
            ..Self::new(text)
        }
    }
}

impl Generator for Fake {
    fn load(&mut self, model: &Path) -> Result<(), String> {
        if model.to_str() == Some("missing.gguf") {
            return Err("no such file".into());
        }
        self.loads.fetch_add(1, Ordering::SeqCst);
        Ok(())
    }
    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        n_ctx: u32,
        _grammar: Option<&str>,
        on_progress: &mut dyn FnMut(Phase, u32, u32) -> bool,
    ) -> Result<Generated, String> {
        self.contexts.lock().unwrap().push(n_ctx);
        if prompt == "explode" {
            return Err("out of memory".into());
        }
        let mut text = String::new();
        let total = (self.text.chars().count() as u32).min(max_tokens);
        for (i, c) in self.text.chars().take(total as usize).enumerate() {
            std::thread::sleep(self.delay);
            text.push(c);
            if !on_progress(Phase::Generate, i as u32 + 1, total) {
                return Err("cancelled".into());
            }
        }
        Ok(Generated {
            text,
            prompt_tokens: prompt.len() as u32,
            gen_tokens: total,
            ..Generated::default()
        })
    }
}

/// Feeds lines to the server as if over a pipe; dropping it closes stdin.
struct Pipe(mpsc::Receiver<Vec<u8>>, Vec<u8>);

impl Read for Pipe {
    fn read(&mut self, buf: &mut [u8]) -> std::io::Result<usize> {
        if self.1.is_empty() {
            match self.0.recv() {
                Ok(bytes) => self.1 = bytes,
                Err(_) => return Ok(0), // the writer is gone: end of input
            }
        }
        let n = buf.len().min(self.1.len());
        buf[..n].copy_from_slice(&self.1[..n]);
        self.1.drain(..n);
        Ok(n)
    }
}

#[derive(Clone, Default)]
struct Out(Arc<Mutex<Vec<u8>>>);

impl Write for Out {
    fn write(&mut self, buf: &[u8]) -> std::io::Result<usize> {
        self.0.lock().unwrap().extend_from_slice(buf);
        Ok(buf.len())
    }
    fn flush(&mut self) -> std::io::Result<()> {
        Ok(())
    }
}

impl Out {
    fn events(&self) -> Vec<Value> {
        let bytes = self.0.lock().unwrap().clone();
        String::from_utf8(bytes)
            .unwrap()
            .lines()
            .map(|l| serde_json::from_str(l).unwrap_or_else(|e| panic!("not JSON ({e}): {l}")))
            .collect()
    }
    fn wait_for(&self, kind: &str, id: &str) -> Value {
        let deadline = Instant::now() + Duration::from_secs(5);
        loop {
            if let Some(e) = self
                .events()
                .into_iter()
                .find(|e| e["type"] == kind && e["id"] == id)
            {
                return e;
            }
            assert!(
                Instant::now() < deadline,
                "no {kind} for {id}: {:?}",
                self.events()
            );
            std::thread::sleep(Duration::from_millis(5));
        }
    }
}

struct Running {
    tx: Option<mpsc::Sender<Vec<u8>>>,
    out: Out,
    done: mpsc::Receiver<()>,
}

impl Running {
    fn send(&self, v: Value) {
        self.raw(&format!("{v}\n"));
    }
    fn raw(&self, line: &str) {
        self.tx
            .as_ref()
            .unwrap()
            .send(line.as_bytes().to_vec())
            .unwrap();
    }
    fn close_stdin(&mut self) {
        self.tx.take();
    }
    fn finished_within(&self, ms: u64) -> bool {
        self.done.recv_timeout(Duration::from_millis(ms)).is_ok()
    }
}

fn start(gen: Fake) -> Running {
    let (tx, rx) = mpsc::channel();
    let out = Out::default();
    let (done_tx, done) = mpsc::channel();
    let writer = out.clone();
    std::thread::spawn(move || {
        serve(BufReader::new(Pipe(rx, Vec::new())), writer, gen);
        let _ = done_tx.send(());
    });
    Running {
        tx: Some(tx),
        out,
        done,
    }
}

fn generate(id: &str, prompt: &str) -> Value {
    json!({"type": "generate", "id": id, "model": "m.gguf", "prompt": prompt, "max_tokens": 256, "n_ctx": 4096})
}

#[test]
fn a_request_loads_the_model_reports_progress_and_returns_the_text() {
    let r = start(Fake::new("요약입니다"));
    r.send(generate("a", "회의 전사"));

    let done = r.out.wait_for("done", "a");

    assert_eq!(done["text"], "요약입니다");
    assert_eq!(done["stats"]["gen_tokens"], 5);
    assert!(done["stats"]["load_sec"].as_f64().unwrap() >= 0.0);
    let events = r.out.events();
    let kinds: Vec<&str> = events.iter().map(|e| e["type"].as_str().unwrap()).collect();
    assert_eq!(kinds.first(), Some(&"loaded"));
    let progress: Vec<u64> = events
        .iter()
        .filter(|e| e["type"] == "progress")
        .map(|e| e["done"].as_u64().unwrap())
        .collect();
    assert!(
        !progress.is_empty() && progress.windows(2).all(|w| w[0] <= w[1]),
        "{progress:?}"
    );
    assert!(events
        .iter()
        .filter(|e| e["type"] == "progress")
        .all(|e| e["phase"] == "generate"));
}

#[test]
fn the_model_is_loaded_once_for_many_requests() {
    let fake = Fake::new("ok");
    let loads = fake.loads.clone();
    let r = start(fake);
    r.send(generate("a", "x"));
    r.send(generate("b", "y"));

    r.out.wait_for("done", "b");

    assert_eq!(loads.load(Ordering::SeqCst), 1);
    assert_eq!(
        r.out
            .events()
            .iter()
            .filter(|e| e["type"] == "loaded")
            .count(),
        1
    );
}

#[test]
fn requests_are_answered_in_order() {
    let r = start(Fake::slow("abc", 5));
    r.send(generate("first", "x"));
    r.send(generate("second", "y"));

    r.out.wait_for("done", "second");

    let ids: Vec<String> = r
        .out
        .events()
        .iter()
        .filter(|e| e["type"] == "done")
        .map(|e| e["id"].as_str().unwrap().to_string())
        .collect();
    assert_eq!(ids, ["first", "second"]);
}

#[test]
fn cancelling_stops_a_running_generation() {
    let r = start(Fake::slow("이 문장은 아주 천천히 하나씩 나옵니다", 50));
    r.send(generate("slow", "x"));
    r.out.wait_for("progress", "slow");

    r.send(json!({"type": "cancel", "id": "slow"}));

    let err = r.out.wait_for("error", "slow");
    assert_eq!(err["kind"], "cancelled");
    assert!(r.out.events().iter().all(|e| e["type"] != "done"));
}

#[test]
fn a_cancelled_request_still_waiting_never_runs() {
    let r = start(Fake::slow("abcdefghij", 30));
    r.send(generate("busy", "x"));
    r.send(generate("queued", "y"));
    r.send(json!({"type": "cancel", "id": "queued"}));

    r.out.wait_for("done", "busy");
    let err = r.out.wait_for("error", "queued");

    assert_eq!(err["kind"], "cancelled");
    assert!(r
        .out
        .events()
        .iter()
        .all(|e| !(e["type"] == "progress" && e["id"] == "queued")));
}

#[test]
fn a_bad_line_is_answered_with_an_error_and_the_loop_carries_on() {
    let r = start(Fake::new("ok"));
    r.raw("this is not json\n");
    r.send(json!({"type": "reticulate", "id": "x"}));
    r.send(json!({"type": "generate", "id": "half"})); // fields missing
    r.send(generate("good", "x"));

    r.out.wait_for("done", "good");

    let errors: Vec<Value> = r
        .out
        .events()
        .into_iter()
        .filter(|e| e["type"] == "error")
        .collect();
    assert_eq!(errors.len(), 3, "{errors:?}");
    assert!(errors.iter().all(|e| e["kind"] == "bad_request"));
    assert_eq!(
        errors[2]["id"], "half",
        "the id is echoed when it can be read"
    );
}

#[test]
fn a_model_that_will_not_load_is_an_error_not_a_crash() {
    let r = start(Fake::new("ok"));
    r.send(json!({"type": "generate", "id": "a", "model": "missing.gguf", "prompt": "x", "max_tokens": 8, "n_ctx": 512}));
    r.send(generate("b", "x"));

    assert_eq!(r.out.wait_for("error", "a")["kind"], "model");
    r.out.wait_for("done", "b");
}

#[test]
fn a_failing_generation_is_an_error_and_the_next_request_still_runs() {
    let r = start(Fake::new("ok"));
    r.send(generate("a", "explode"));
    r.send(generate("b", "x"));

    let err = r.out.wait_for("error", "a");
    assert_eq!(err["kind"], "generate");
    assert!(err["message"].as_str().unwrap().contains("out of memory"));
    r.out.wait_for("done", "b");
}

#[test]
fn closing_stdin_ends_the_loop() {
    // the app quit or died: its end of the pipe closes, and the sidecar must not linger
    let mut r = start(Fake::new("ok"));
    r.close_stdin();
    assert!(r.finished_within(2000));
}

#[test]
fn closing_stdin_mid_generation_stops_it_and_ends_the_loop() {
    let mut r = start(Fake::slow("이 문장은 아주 천천히 하나씩 나옵니다", 100));
    r.send(generate("slow", "x"));
    r.out.wait_for("progress", "slow");

    r.close_stdin();

    assert!(
        r.finished_within(2000),
        "still generating for an app that is gone"
    );
}

#[test]
fn shutdown_ends_the_loop() {
    let r = start(Fake::new("ok"));
    r.send(json!({"type": "shutdown"}));
    assert!(r.finished_within(2000));
}

#[test]
fn every_event_is_one_json_line_of_a_known_type() {
    let r = start(Fake::new("ok"));
    r.raw("{broken\n");
    r.send(generate("a", "x"));
    r.out.wait_for("done", "a");

    for e in r.out.events() {
        let parsed: Event =
            serde_json::from_value(e.clone()).unwrap_or_else(|err| panic!("{err}: {e}"));
        if let Event::Progress { phase, .. } = parsed {
            assert_eq!(phase, Phase::Generate);
        }
    }
}

#[test]
fn max_tokens_is_passed_to_the_generator() {
    let r = start(Fake::new("abcdefghij"));
    r.send(json!({"type": "generate", "id": "a", "model": "m.gguf", "prompt": "x", "max_tokens": 3, "n_ctx": 512}));
    assert_eq!(r.out.wait_for("done", "a")["text"], "abc");
}

#[test]
fn each_request_gets_the_context_it_asks_for_without_a_reload() {
    // a long meeting needs a bigger context than a short one; the model stays loaded
    let fake = Fake::new("ok");
    let (loads, contexts) = (fake.loads.clone(), fake.contexts.clone());
    let r = start(fake);
    r.send(json!({"type": "generate", "id": "a", "model": "m.gguf", "prompt": "x", "max_tokens": 8, "n_ctx": 2048}));
    r.send(json!({"type": "generate", "id": "b", "model": "m.gguf", "prompt": "x", "max_tokens": 8, "n_ctx": 32768}));

    r.out.wait_for("done", "b");

    assert_eq!(*contexts.lock().unwrap(), [2048, 32768]);
    assert_eq!(loads.load(Ordering::SeqCst), 1);
}
