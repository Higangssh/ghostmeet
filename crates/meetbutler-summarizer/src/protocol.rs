//! The wire format: one JSON object per line, each way.
//!
//! App to sidecar: `generate`, `cancel`, `shutdown`. Sidecar to app: `loaded` (once per
//! model), `progress`, then `done` or `error` for every `generate`, in the order asked.

use std::path::PathBuf;

use serde::{Deserialize, Serialize};

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Request {
    Generate(GenerateRequest),
    Cancel { id: String },
    Shutdown,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct GenerateRequest {
    /// Echoed on every event about this request.
    pub id: String,
    /// A GGUF file. Loaded on first use and kept until a different one is asked for.
    pub model: PathBuf,
    /// The whole prompt, chat template already applied.
    pub prompt: String,
    pub max_tokens: u32,
    /// Context size to load the model with.
    pub n_ctx: u32,
    /// A GBNF grammar the output must follow (the fixed summary format), if any.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub grammar: Option<String>,
    /// Penalise repeating the last 256 tokens (llama.cpp's repeat penalty, 1.0 = off).
    /// Small models can otherwise loop on one list item until max_tokens.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub repeat_penalty: Option<f32>,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum Phase {
    /// Reading the prompt.
    Prompt,
    /// Writing the answer.
    Generate,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum ErrorKind {
    /// The line could not be read as a request.
    BadRequest,
    /// The model would not load.
    Model,
    /// Generation failed.
    Generate,
    /// Cancelled before or while running.
    Cancelled,
}

/// What a generator produced, with its timings.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Generated {
    pub text: String,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub prompt_sec: f64,
    pub gen_sec: f64,
}

#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct Stats {
    /// Loading the model for this request; 0 when it was already loaded.
    pub load_sec: f64,
    pub prompt_tokens: u32,
    pub gen_tokens: u32,
    pub prompt_sec: f64,
    pub gen_sec: f64,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "type", rename_all = "snake_case")]
pub enum Event {
    Loaded {
        id: String,
        model: PathBuf,
        sec: f64,
    },
    Progress {
        id: String,
        phase: Phase,
        done: u32,
        total: u32,
    },
    Done {
        id: String,
        text: String,
        stats: Stats,
    },
    Error {
        id: Option<String>,
        kind: ErrorKind,
        message: String,
    },
}
