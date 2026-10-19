//! whisper.cpp through whisper-rs: the decoder.
//!
//! Decoding options are the benchmark's (`bench/engines.py`, `WhisperCpp`): beam search
//! of 5, no text carried between calls, temperature 0 with whisper.cpp's default
//! fallback, 4 threads, flash attention on (whisper-server's default). Speech is cut
//! before it gets here (`silero`, `spans`); whisper.cpp's own VAD is never used.

use std::path::Path;
use std::sync::Arc;

use whisper_rs::{
    FullParams, SamplingStrategy, WhisperContext, WhisperContextParameters, WhisperState,
};

use crate::incremental::RawSegment;
use crate::span_decoder::SpanDecode;
use crate::TranscribeError;

/// Performance cores on an M4; the budget every engine had in the benchmark.
pub const THREADS: i32 = 4;

/// Text context for the vocabulary prompt: whisper's own cap, half its 448-token window.
const PROMPT_CTX: i32 = 224;

/// Route whisper.cpp's and ggml's logging through the `log` crate instead of stderr.
pub fn install_logging() {
    whisper_rs::install_logging_hooks();
}

/// A loaded ggml Whisper model. Load once per process (`ModelCache`) and share it.
///
/// Drop every model before the process exits: with Metal, a model still alive at exit
/// trips ggml's device teardown (`GGML_ASSERT` in `ggml_metal_device_free`) and the
/// process aborts. Keep models out of statics.
pub struct WhisperModel {
    ctx: WhisperContext,
}

impl WhisperModel {
    pub fn load(path: impl AsRef<Path>, gpu: bool) -> Result<Self, TranscribeError> {
        let path = path.as_ref();
        let params = WhisperContextParameters {
            use_gpu: gpu,
            flash_attn: gpu,
            ..Default::default()
        };
        let path_str = path.to_str().ok_or("model path is not UTF-8")?;
        let ctx = WhisperContext::new_with_params(path_str, params)
            .map_err(|e| format!("loading {}: {e}", path.display()))?;
        Ok(Self { ctx })
    }
}

/// One session channel's decoder over a shared model: its own state, its language.
pub struct WhisperDecoder {
    // the state borrows nothing, but must not outlive the model it was made from
    state: WhisperState,
    _model: Arc<WhisperModel>,
    language: Option<String>,
    prompt: Option<String>,
    vocabulary: Vec<String>,
}

impl WhisperDecoder {
    /// `language`: an ISO code such as "ko", or `None` to detect it per span.
    pub fn new(
        model: Arc<WhisperModel>,
        language: Option<String>,
    ) -> Result<Self, TranscribeError> {
        let state = model.ctx.create_state()?;
        Ok(Self {
            state,
            _model: model,
            language,
            prompt: None,
            vocabulary: Vec::new(),
        })
    }

    /// Give every span the user's vocabulary as whisper's initial prompt (STT-7).
    /// With no terms the decoder stays as `new` made it.
    pub fn with_vocabulary(mut self, terms: &[String]) -> Self {
        self.prompt = crate::vocabulary::prompt(terms);
        self.vocabulary = crate::vocabulary::kept(terms);
        self
    }
}

impl SpanDecode for WhisperDecoder {
    fn vocabulary(&self) -> &[String] {
        &self.vocabulary
    }

    fn decode(&mut self, audio: &[f32]) -> Result<Vec<RawSegment>, TranscribeError> {
        let mut params = FullParams::new(SamplingStrategy::BeamSearch {
            beam_size: 5,
            patience: -1.0,
        });
        params.set_language(Some(self.language.as_deref().unwrap_or("auto")));
        params.set_n_threads(THREADS);
        // each span is decoded on its own; carrying text across lets a hallucinated
        // phrase repeat itself for the rest of the meeting
        params.set_no_context(true);
        match &self.prompt {
            // no_context clears the past, so each span starts from the vocabulary alone;
            // whisper.cpp drops any prompt when the text context is 0
            Some(prompt) => {
                params.set_initial_prompt(prompt);
                params.set_n_max_text_ctx(PROMPT_CTX);
            }
            None => params.set_n_max_text_ctx(0),
        }
        params.set_temperature(0.0);
        params.set_print_special(false);
        params.set_print_progress(false);
        params.set_print_realtime(false);
        params.set_print_timestamps(false);

        self.state.full(params, audio)?;
        let mut out = Vec::new();
        for segment in self.state.as_iter() {
            out.push(RawSegment {
                start: segment.start_timestamp() as f64 / 100.0,
                end: segment.end_timestamp() as f64 / 100.0,
                text: segment.to_str_lossy()?.into_owned(),
            });
        }
        Ok(out)
    }
}
