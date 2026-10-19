//! Silero VAD as the benchmark ran it: k2-fsa's onnx export (`silero_vad.onnx`, MIT),
//! 512-sample frames with the LSTM state carried from frame to frame, on tract, a
//! pure-Rust ONNX runtime (no native library, no network).
//!
//! Not whisper.cpp's ggml Silero v5.1.2: that port reflect-pads each frame instead of
//! carrying the previous frame's 64 samples as context, and on FLEURS it found 18% less
//! speech than this model and dropped whole utterances (`bench/README.md`).

use std::path::Path;

use tract_onnx::prelude::*;

use crate::span_decoder::SpeechProbs;
use crate::spans::FRAME;
use crate::TranscribeError;

type Plan = std::sync::Arc<TypedSimplePlan>;

pub struct SileroOnnx {
    plan: Plan,
}

fn state() -> Tensor {
    tract_ndarray::Array3::<f32>::zeros((2, 1, 64)).into()
}

impl SileroOnnx {
    pub fn load(path: impl AsRef<Path>) -> Result<Self, TranscribeError> {
        let path = path.as_ref();
        let plan = tract_onnx::onnx()
            .model_for_path(path)
            .and_then(|m| m.with_input_fact(0, f32::fact([1, FRAME]).into()))
            .and_then(|m| m.with_input_fact(1, f32::fact([2, 1, 64]).into()))
            .and_then(|m| m.with_input_fact(2, f32::fact([2, 1, 64]).into()))
            .and_then(|m| m.into_optimized())
            .and_then(|m| m.into_runnable())
            .map_err(|e| format!("loading {}: {e}", path.display()))?;
        Ok(Self { plan })
    }
}

impl SpeechProbs for SileroOnnx {
    /// One probability per whole frame of `audio`, starting from a fresh state, as the
    /// benchmark did for every window it cut.
    fn probs(&mut self, audio: &[f32]) -> Result<Vec<f32>, TranscribeError> {
        let (mut h, mut c) = (state(), state());
        let mut out = Vec::with_capacity(audio.len() / FRAME);
        for frame in audio.chunks_exact(FRAME) {
            let x: Tensor =
                tract_ndarray::Array2::from_shape_vec((1, FRAME), frame.to_vec())?.into();
            let mut result = self.plan.run(tvec!(x.into(), h.into(), c.into()))?;
            c = result.remove(2).into_tensor();
            h = result.remove(1).into_tensor();
            let prob = result[0]
                .to_plain_array_view::<f32>()?
                .iter()
                .next()
                .copied()
                .ok_or("empty output")?;
            out.push(prob);
        }
        Ok(out)
    }
}
