//! The app's side of the sidecar: start it, send requests, read events, and make sure it
//! never outlives the app.
//!
//! The child's stdin is the app's lifeline to it: the sidecar exits when that pipe
//! closes, so even an app killed outright leaves no process behind. Dropping a `Sidecar`
//! also kills the child outright.

use std::io::{self, BufRead, BufReader, Write};
use std::path::Path;
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc::{self, Receiver};
use std::sync::{Arc, Mutex};

use crate::protocol::{ErrorKind, Event, GenerateRequest, Request, Stats};

#[derive(Debug, thiserror::Error)]
pub enum ClientError {
    #[error("the summarizer exited ({0:?})")]
    Exited(Option<i32>),
    #[error("{kind:?}: {message}")]
    Failed { kind: ErrorKind, message: String },
    #[error(transparent)]
    Io(#[from] io::Error),
}

pub struct Sidecar {
    child: Child,
    stdin: Arc<Mutex<Option<ChildStdin>>>,
    events: Receiver<Event>,
    next: u64,
}

/// Cancels requests from another thread.
#[derive(Clone)]
pub struct Canceller(Arc<Mutex<Option<ChildStdin>>>);

impl Canceller {
    pub fn cancel(&self, id: &str) {
        let _ = send(&self.0, &Request::Cancel { id: id.to_string() });
    }
}

fn send(stdin: &Mutex<Option<ChildStdin>>, request: &Request) -> io::Result<()> {
    let mut guard = stdin.lock().unwrap_or_else(|e| e.into_inner());
    let pipe = guard
        .as_mut()
        .ok_or_else(|| io::Error::new(io::ErrorKind::BrokenPipe, "closed"))?;
    let line = serde_json::to_string(request).map_err(io::Error::other)?;
    writeln!(pipe, "{line}")?;
    pipe.flush()
}

impl Sidecar {
    /// Start `program` with `args`. Its stderr (llama.cpp's logging) goes to `log`.
    pub fn spawn(program: &Path, args: &[&str]) -> io::Result<Self> {
        let mut child = Command::new(program)
            .args(args)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::piped())
            .spawn()?;
        let stdout = child.stdout.take().expect("piped");
        let stderr = child.stderr.take().expect("piped");
        let (tx, events) = mpsc::channel();
        std::thread::spawn(move || {
            for line in BufReader::new(stdout).lines() {
                let Ok(line) = line else { break };
                match serde_json::from_str::<Event>(&line) {
                    Ok(event) => {
                        if tx.send(event).is_err() {
                            break;
                        }
                    }
                    Err(e) => log::warn!("summarizer said something unreadable ({e})"),
                }
            }
            // dropping tx tells the waiting caller the process is gone
        });
        std::thread::spawn(move || {
            for line in BufReader::new(stderr).lines().map_while(Result::ok) {
                log::debug!("summarizer: {line}");
            }
        });
        let stdin = Arc::new(Mutex::new(child.stdin.take()));
        Ok(Self {
            child,
            stdin,
            events,
            next: 0,
        })
    }

    pub fn pid(&self) -> u32 {
        self.child.id()
    }

    pub fn canceller(&self) -> Canceller {
        Canceller(self.stdin.clone())
    }

    /// Run one request to the end. `on_event` sees every event about it (`loaded`,
    /// `progress`) as it arrives. The request's id is assigned here.
    pub fn generate(
        &mut self,
        mut request: GenerateRequest,
        mut on_event: impl FnMut(&Event),
    ) -> Result<(String, Stats), ClientError> {
        self.next += 1;
        request.id = format!("r{}", self.next);
        let id = request.id.clone();
        if send(&self.stdin, &Request::Generate(request)).is_err() {
            return Err(self.exited());
        }
        loop {
            let Ok(event) = self.events.recv() else {
                return Err(self.exited());
            };
            let ours = match &event {
                Event::Loaded { id: i, .. }
                | Event::Progress { id: i, .. }
                | Event::Done { id: i, .. } => i == &id,
                Event::Error { id: i, .. } => i.as_deref() == Some(id.as_str()),
            };
            if !ours {
                continue; // a late event about an earlier, abandoned request
            }
            match event {
                Event::Done { text, stats, .. } => return Ok((text, stats)),
                Event::Error { kind, message, .. } => {
                    return Err(ClientError::Failed { kind, message })
                }
                other => on_event(&other),
            }
        }
    }

    fn exited(&mut self) -> ClientError {
        let code = self.child.wait().ok().and_then(|s| s.code());
        ClientError::Exited(code)
    }
}

impl Drop for Sidecar {
    fn drop(&mut self) {
        // closing stdin is enough for a healthy sidecar; kill covers one that is stuck
        self.stdin.lock().unwrap_or_else(|e| e.into_inner()).take();
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}
