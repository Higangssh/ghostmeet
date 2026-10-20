#!/usr/bin/env bash
# Build the app bundle with the summary sidecar inside it, next to the app's executable.
#
#     app/bundle.sh
#
# The sidecar is built with llama.cpp (`--features llama`) and copied to where Tauri's
# externalBin looks for it, named for the host target. The main tauri.conf.json leaves
# externalBin out so `cargo build` and `cargo test` work without the sidecar built;
# tauri.bundle.json adds it for the bundle only. Needs the Tauri CLI (`cargo tauri`).
set -euo pipefail
cd "$(dirname "$0")/.."

cargo build --release -p meetbutler-summarizer --features llama --bin meetbutler-summarizer
triple=$(rustc -vV | sed -n 's/^host: //p')
mkdir -p app/src-tauri/binaries
cp target/release/meetbutler-summarizer "app/src-tauri/binaries/meetbutler-summarizer-$triple"
cd app/src-tauri
cargo tauri build --config tauri.bundle.json
