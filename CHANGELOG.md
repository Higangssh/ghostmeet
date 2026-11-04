# Changelog

All notable changes to this project. Versions follow SemVer; before 1.0 a minor is a
feature batch and a patch is a fix (`requirements.md` §9).

## [Unreleased] — 0.2.0, the first desktop version

ghostmeet becomes MeetButler: a Tauri 2 desktop app that records the microphone and
system audio separately and transcribes and summarises on the machine. The v0.1 Chrome
extension and Python backend move to `legacy/` (tag `v0.1.0`).

### Added

- **Recording** (REC-1, REC-2): microphone (cpal) and system audio (ScreenCaptureKit on
  macOS; WASAPI loopback through cpal on Windows, built but not yet run on a real
  machine, `crates/meetbutler-audio/WINDOWS.md`), stored per channel as `me` / `them`.
  Records with whatever source opens and reports the rest. Capture is supervised: a
  stream that stops delivering is reopened, and a hole in the audio is filled with
  silence so the timeline stays true.
- **Live captions** (STT-1~3): whisper.cpp `small` on speech spans cut by Silero v4 onnx
  on tract. Each pass reads a bounded tail window of the audio on disk, so its cost stays
  flat in 4-hour meetings. Segments are saved as they are recognised; a session a dead
  process left behind is marked interrupted on the next start.
- **Final pass** (STT-6): after Stop, `large-v3-turbo` transcribes the meeting again span
  by span and replaces the transcript in one transaction. A failed or cancelled pass
  leaves the live captions as they were.
- **Vocabulary** (STT-7): names and terms given to transcription as a hint, carried over
  from the last meeting.
- **Hallucination blocklist**: drops a span only when it is under 2 s and holds a known
  subtitle credit; short real replies survive.
- **Speakers by channel** (SPK-1): me / them.
- **Summary** (SUM-1~3): Qwen3.5-4B in a separate sidecar executable
  (`meetbutler-summarizer`, llama.cpp) that talks over stdin/stdout, summarises in parts
  and merges. Summary, decisions and action items, each linked to the lines it rests on.
  Cancel and retry from the panel.
- **Windows** (MEM-1, OUT-1): mini bar, panel (captions / summary / record tabs) and
  library (past meetings, rename, delete), Markdown copy and save to Downloads.
- **Menu bar** (REC-3, in part): start/stop, panel, library and quit from the tray icon,
  which shows when a meeting is recording.
- **Bundle**: `app/bundle.sh` builds the .app and .dmg with the sidecar inside, signed ad
  hoc with the hardened runtime and the microphone entitlement.
- **README** rewritten for MeetButler, with a demo and screenshots made from the app's
  pages playing an invented meeting (`app/demo-capture/`); the v0.1 guide moves to
  `legacy/README.md`.
- **Benchmarks** (`bench/`): engine, VAD, summary model and diarization measured on public
  speech (FLEURS, AMI) and synthetic audio only.

### Security

- The app opens no port; the windows talk to Rust over Tauri IPC under a strict CSP and
  the core permissions only. `app/src-tauri/tests/security.rs` checks this, and that no
  server, HTTP client or telemetry crate is built for a desktop target, the sidecar
  included.
- The app never downloads models; they are put in the models folder by hand.

### Not yet in 0.2

- Global shortcut to start and stop (the rest of REC-3).
- Windows system audio checked on a real machine (`app/README.md`, manual checks §8).
- Running the final pass on a session a hard exit left interrupted (`app/README.md`,
  backlog).
- Model choice and download management (STT-5).
- Developer ID signing and notarisation: a downloaded copy is stopped by Gatekeeper.
