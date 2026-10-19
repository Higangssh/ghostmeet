//! A stand-in model for tests of the process boundary (`--backend fake`): answers with
//! a fixed summary, one character per token, optionally slowly. It loads nothing.

use std::path::Path;
use std::time::Duration;

use crate::protocol::{Generated, Phase};
use crate::server::Generator;

pub const ANSWER: &str =
    r#"{"summary":"가짜 요약","decisions":[],"action_items":[],"open_questions":[]}"#;

pub struct Fake {
    pub delay: Duration,
}

impl Generator for Fake {
    fn load(&mut self, model: &Path) -> Result<(), String> {
        if model.as_os_str().is_empty() {
            return Err("no model given".into());
        }
        Ok(())
    }

    fn generate(
        &mut self,
        prompt: &str,
        max_tokens: u32,
        _n_ctx: u32,
        _grammar: Option<&str>,
        on_progress: &mut dyn FnMut(Phase, u32, u32) -> bool,
    ) -> Result<Generated, String> {
        let total = (ANSWER.chars().count() as u32).min(max_tokens);
        let mut text = String::new();
        for (i, c) in ANSWER.chars().take(total as usize).enumerate() {
            std::thread::sleep(self.delay);
            text.push(c);
            if !on_progress(Phase::Generate, i as u32 + 1, total) {
                break;
            }
        }
        Ok(Generated {
            text,
            prompt_tokens: prompt.chars().count() as u32,
            gen_tokens: total,
            ..Generated::default()
        })
    }
}
