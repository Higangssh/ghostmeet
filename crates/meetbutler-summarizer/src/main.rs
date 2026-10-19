//! meetbutler-summarizer: JSON lines on stdin, JSON lines on stdout, logs on stderr.
//!
//!     meetbutler-summarizer [--backend llama|fake] [--fake-delay-ms N]

use std::io::{stdin, stdout, BufReader};
use std::time::Duration;

use meetbutler_summarizer::fake::Fake;
use meetbutler_summarizer::server::serve;

fn main() {
    let args: Vec<String> = std::env::args().skip(1).collect();
    let value = |flag: &str| {
        args.iter()
            .position(|a| a == flag)
            .and_then(|i| args.get(i + 1))
            .cloned()
    };
    let backend = value("--backend").unwrap_or_else(|| "llama".into());
    let input = BufReader::new(stdin());

    match backend.as_str() {
        "fake" => {
            let ms = value("--fake-delay-ms")
                .and_then(|v| v.parse().ok())
                .unwrap_or(0);
            serve(
                input,
                stdout(),
                Fake {
                    delay: Duration::from_millis(ms),
                },
            );
        }
        #[cfg(feature = "llama")]
        "llama" => serve(input, stdout(), meetbutler_summarizer::llama::Llama::new()),
        other => {
            eprintln!("meetbutler-summarizer: backend {other} is not built in");
            std::process::exit(2);
        }
    }
}
