//! The process boundary: the app-side client against the real sidecar executable, with
//! the fake backend so no model is needed.

use std::io::{BufRead, BufReader, Write};
use std::process::{Command, Stdio};
use std::sync::mpsc;
use std::time::{Duration, Instant};

use meetbutler_summarizer::client::{ClientError, Sidecar};
use meetbutler_summarizer::fake::ANSWER;
use meetbutler_summarizer::protocol::{ErrorKind, Event, GenerateRequest};

const BIN: &str = env!("CARGO_BIN_EXE_meetbutler-summarizer");

fn spawn(delay_ms: u64) -> Sidecar {
    Sidecar::spawn(
        BIN.as_ref(),
        &[
            "--backend",
            "fake",
            "--fake-delay-ms",
            &delay_ms.to_string(),
        ],
    )
    .unwrap()
}

fn request(prompt: &str) -> GenerateRequest {
    GenerateRequest {
        id: String::new(), // the client assigns one
        model: "m.gguf".into(),
        prompt: prompt.into(),
        max_tokens: 512,
        n_ctx: 4096,
        grammar: None,
        repeat_penalty: None,
    }
}

fn alive(pid: u32) -> bool {
    Command::new("kill")
        .args(["-0", &pid.to_string()])
        .stderr(Stdio::null())
        .status()
        .unwrap()
        .success()
}

fn gone_within(pid: u32, ms: u64) -> bool {
    let deadline = Instant::now() + Duration::from_millis(ms);
    while Instant::now() < deadline {
        if !alive(pid) {
            return true;
        }
        std::thread::sleep(Duration::from_millis(20));
    }
    !alive(pid)
}

#[test]
fn a_request_comes_back_with_text_and_stats() {
    let mut s = spawn(0);
    let (text, stats) = s.generate(request("회의"), |_| {}).unwrap();
    assert_eq!(text, ANSWER);
    assert_eq!(stats.gen_tokens as usize, ANSWER.chars().count());
}

#[test]
fn progress_reaches_the_caller() {
    let mut s = spawn(0);
    let mut seen = Vec::new();
    s.generate(request("x"), |e| seen.push(e.clone())).unwrap();
    assert!(seen.iter().any(|e| matches!(e, Event::Loaded { .. })));
    assert!(seen.iter().any(|e| matches!(e, Event::Progress { .. })));
}

#[test]
fn several_requests_share_one_process() {
    let mut s = spawn(0);
    let pid = s.pid();
    for _ in 0..3 {
        s.generate(request("x"), |_| {}).unwrap();
    }
    assert_eq!(s.pid(), pid);
}

#[test]
fn a_request_can_be_cancelled_from_another_thread() {
    let mut s = spawn(30);
    let canceller = s.canceller();
    let (tx, rx) = mpsc::channel();
    let started = Instant::now();
    let result = s.generate(request("x"), |e| {
        if let Event::Progress { id, .. } = e {
            let _ = tx.send(id.clone());
        }
        if let Ok(id) = rx.try_recv() {
            canceller.cancel(&id);
        }
    });
    assert!(
        matches!(
            result,
            Err(ClientError::Failed {
                kind: ErrorKind::Cancelled,
                ..
            })
        ),
        "{result:?}"
    );
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "the whole answer would take ~2.4 s"
    );
}

#[test]
fn a_sidecar_that_dies_mid_request_is_an_error_not_a_hang() {
    let mut s = spawn(50);
    let pid = s.pid();
    let result = s.generate(request("x"), |e| {
        if matches!(e, Event::Progress { .. }) {
            let _ = Command::new("kill").args(["-9", &pid.to_string()]).status();
        }
    });
    assert!(matches!(result, Err(ClientError::Exited(_))), "{result:?}");
}

#[test]
fn a_dead_sidecar_refuses_further_requests_promptly() {
    let mut s = spawn(0);
    let _ = Command::new("kill")
        .args(["-9", &s.pid().to_string()])
        .status();
    std::thread::sleep(Duration::from_millis(200)); // dead, though unreaped until the client waits
    let started = Instant::now();
    assert!(s.generate(request("x"), |_| {}).is_err());
    assert!(started.elapsed() < Duration::from_secs(2));
}

#[test]
fn dropping_the_client_ends_the_sidecar() {
    let s = spawn(50);
    let pid = s.pid();
    drop(s);
    assert!(gone_within(pid, 2000), "the sidecar outlived its client");
}

#[test]
fn the_sidecar_exits_by_itself_when_the_app_goes_away() {
    // if the app is killed outright nobody calls drop; its end of stdin closes, and that
    // alone must end the sidecar, even mid-generation
    let mut child = Command::new(BIN)
        .args(["--backend", "fake", "--fake-delay-ms", "100"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    let mut stdin = child.stdin.take().unwrap();
    writeln!(stdin, r#"{{"type":"generate","id":"a","model":"m.gguf","prompt":"x","max_tokens":512,"n_ctx":512}}"#)
        .unwrap();
    let mut out = BufReader::new(child.stdout.take().unwrap());
    let mut line = String::new();
    out.read_line(&mut line).unwrap(); // loaded: it is running
    drop(stdin);

    let deadline = Instant::now() + Duration::from_secs(3);
    loop {
        if let Some(status) = child.try_wait().unwrap() {
            assert!(status.success(), "{status}");
            break;
        }
        assert!(
            Instant::now() < deadline,
            "still running with nobody listening"
        );
        std::thread::sleep(Duration::from_millis(20));
    }
}

#[test]
fn stdout_carries_protocol_lines_only() {
    let mut child = Command::new(BIN)
        .args(["--backend", "fake"])
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()
        .unwrap();
    {
        let mut stdin = child.stdin.take().unwrap();
        writeln!(stdin, r#"{{"type":"generate","id":"a","model":"m.gguf","prompt":"x","max_tokens":512,"n_ctx":512}}"#)
            .unwrap();
        writeln!(stdin, "nonsense").unwrap();
        writeln!(stdin, r#"{{"type":"shutdown"}}"#).unwrap();
    }
    let out = child.wait_with_output().unwrap();
    let text = String::from_utf8(out.stdout).unwrap();
    assert!(!text.is_empty());
    for line in text.lines() {
        serde_json::from_str::<Event>(line).unwrap_or_else(|e| panic!("{e}: {line}"));
    }
}

#[test]
fn an_unknown_backend_fails_to_start_cleanly() {
    let status = Command::new(BIN)
        .args(["--backend", "nope"])
        .stdin(Stdio::null())
        .stderr(Stdio::null())
        .status()
        .unwrap();
    assert_eq!(status.code(), Some(2));
}

#[test]
fn a_whole_summary_runs_through_the_sidecar_and_stops_when_cancelled() {
    use meetbutler_summarizer::summary::{summarise, Options};
    use std::sync::atomic::{AtomicBool, Ordering};

    let lines: Vec<String> = (0..4).map(|_| "가".repeat(1000)).collect();
    let opts = Options {
        model: "m.gguf".into(),
    };

    let mut sidecar = spawn(0);
    let s = summarise(&mut sidecar, &lines, &opts, &AtomicBool::new(false), |_| {}).unwrap();
    assert_eq!((s.chunks, s.chunks_broken), (2, 0));
    assert_eq!(
        s.summary, ANSWER,
        "the fake answers the overall prompt with its fixed text"
    );

    // slow enough that the cancel lands mid-generation of the first chunk
    let mut sidecar = spawn(20);
    let cancel = AtomicBool::new(false);
    let started = Instant::now();
    let err = summarise(&mut sidecar, &lines, &opts, &cancel, |p| {
        if p > 0.0 {
            cancel.store(true, Ordering::SeqCst);
        }
    })
    .unwrap_err();
    assert!(
        matches!(
            err,
            ClientError::Failed {
                kind: ErrorKind::Cancelled,
                ..
            }
        ),
        "{err:?}"
    );
    // the fake's answer takes about 1.6 s at 20 ms a token; cancelled, far less
    assert!(
        started.elapsed() < Duration::from_millis(1000),
        "{:?}",
        started.elapsed()
    );
}
