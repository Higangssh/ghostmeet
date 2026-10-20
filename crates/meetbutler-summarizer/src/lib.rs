//! MeetButler's summary sidecar.
//!
//! A separate executable the app ships next to itself and talks to over stdin/stdout,
//! one JSON object per line (`protocol`). It exists because llama.cpp and whisper.cpp
//! each bring their own copy of ggml, at different versions, and the two cannot share a
//! process. Being separate also means the LLM's memory comes back when it exits, and a
//! crash in it cannot take a recording down with it.
//!
//! It opens no port, reads nothing but its stdin and the model file it is given, and
//! exits when its stdin closes, so it never outlives the app.

pub mod client;
pub mod fake;
#[cfg(feature = "llama")]
pub mod llama;
pub mod merge;
pub mod protocol;
pub mod server;
pub mod summary;
