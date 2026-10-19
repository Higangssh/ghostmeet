//! llama.cpp through llama-cpp-2. Greedy decoding, so a benchmark run is repeatable;
//! an optional GBNF grammar keeps the output in the fixed summary format.

use std::num::NonZeroU32;
use std::path::Path;
use std::time::Instant;

use llama_cpp_2::context::params::LlamaContextParams;
use llama_cpp_2::llama_backend::LlamaBackend;
use llama_cpp_2::llama_batch::LlamaBatch;
use llama_cpp_2::model::params::LlamaModelParams;
use llama_cpp_2::model::LlamaModel;
use llama_cpp_2::sampling::LlamaSampler;

use crate::protocol::{Generated, Phase};
use crate::server::Generator;

/// Prompt tokens decoded per batch.
const BATCH: usize = 512;
/// Performance cores on an M4; the budget the transcription benchmark used too.
const THREADS: i32 = 4;

pub struct Llama {
    backend: LlamaBackend,
    model: Option<LlamaModel>,
}

impl Llama {
    pub fn new() -> Self {
        let mut backend = LlamaBackend::init().expect("llama.cpp backend initialises once");
        backend.void_logs();
        Self {
            backend,
            model: None,
        }
    }
}

impl Default for Llama {
    fn default() -> Self {
        Self::new()
    }
}

impl Generator for Llama {
    fn load(&mut self, model: &Path) -> Result<(), String> {
        self.model = None; // free the old one first: two at once may not fit
        let params = LlamaModelParams::default().with_n_gpu_layers(u32::MAX);
        let loaded = LlamaModel::load_from_file(&self.backend, model, &params)
            .map_err(|e| format!("loading {}: {e}", model.display()))?;
        self.model = Some(loaded);
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        n_ctx: u32,
        grammar: Option<&str>,
        on_progress: &mut dyn FnMut(Phase, u32, u32) -> bool,
    ) -> Result<Generated, String> {
        let model = self.model.as_ref().ok_or("no model loaded")?;
        let vocab = model.vocab();
        let tokens = vocab.tokenize(prompt.as_bytes(), true, true);
        if tokens.len() as u32 + max_tokens > n_ctx {
            return Err(format!(
                "prompt of {} tokens plus {max_tokens} to generate does not fit a context of {}",
                tokens.len(),
                n_ctx
            ));
        }
        let params = LlamaContextParams::default()
            .with_n_ctx(NonZeroU32::new(n_ctx))
            .with_n_batch(BATCH as u32)
            .with_n_threads(THREADS)
            .with_n_threads_batch(THREADS);
        let mut ctx = model
            .new_context(&self.backend, params)
            .map_err(|e| e.to_string())?;

        // read the prompt
        let started = Instant::now();
        let total = tokens.len() as u32;
        let mut batch = LlamaBatch::new(BATCH, 1);
        for (chunk_no, chunk) in tokens.chunks(BATCH).enumerate() {
            batch.clear();
            let base = chunk_no * BATCH;
            for (i, tok) in chunk.iter().enumerate() {
                let pos = base + i;
                batch
                    .add(*tok, pos as i32, &[0], pos + 1 == tokens.len())
                    .map_err(|e| e.to_string())?;
            }
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
            if !on_progress(Phase::Prompt, (base + chunk.len()) as u32, total) {
                return Err("cancelled".into());
            }
        }
        let prompt_sec = started.elapsed().as_secs_f64();

        // write the answer
        let mut sampler = match grammar {
            Some(g) => LlamaSampler::chain_simple([
                LlamaSampler::grammar(model, g, "root").map_err(|e| format!("grammar: {e}"))?,
                LlamaSampler::greedy(),
            ]),
            None => LlamaSampler::greedy(),
        };
        let started = Instant::now();
        let mut out = Vec::new();
        let mut pos = tokens.len() as i32;
        let mut produced = 0;
        while produced < max_tokens {
            // sample() also accepts the token; accepting again would advance the grammar twice
            let tok = sampler.sample(&ctx, batch.n_tokens() - 1);
            if vocab.is_eog(tok) {
                break;
            }
            out.extend(vocab.token_to_piece(tok, false, None));
            produced += 1;
            if !on_progress(Phase::Generate, produced, max_tokens) {
                return Err("cancelled".into());
            }
            batch.clear();
            batch.add(tok, pos, &[0], true).map_err(|e| e.to_string())?;
            pos += 1;
            ctx.decode(&mut batch).map_err(|e| e.to_string())?;
        }
        Ok(Generated {
            text: String::from_utf8_lossy(&out).into_owned(),
            prompt_tokens: total,
            gen_tokens: produced,
            prompt_sec,
            gen_sec: started.elapsed().as_secs_f64(),
        })
    }
}
