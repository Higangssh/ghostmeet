# MeetButler app (0.2 skeleton)

Tauri 2 shell over `crates/meetbutler-engine` and `crates/meetbutler-audio`. Opens no
port; the window talks to Rust over Tauri IPC only.

```bash
# models are read from MEETBUTLER_MODELS, else <app data>/models; the app never downloads them
cargo build --release -p meetbutler-summarizer --features llama   # the summary sidecar, with llama.cpp
MEETBUTLER_MODELS=$PWD/bench/data/models cargo run -p meetbutler --release
cargo test -p meetbutler                      # security posture (no port, CSP, no network crates)
app/bundle.sh                                 # the .app/.dmg with the sidecar inside (needs `cargo tauri`)
```

The bundle needs the Tauri CLI, a development tool only: `cargo install tauri-cli
--version 2.12.3 --locked` (the version this was built with).

The summary runs in `meetbutler-summarizer`, a separate executable the app starts after
the final pass and that exits when the summary is done. The app looks for it next to its
own executable (`MEETBUTLER_SUMMARIZER` overrides). A workspace `cargo build --release`
without `--features llama` rebuilds it with the test backend only; build it again with
the feature before running the app.

Models, all downloaded ahead of time into that folder:

| File | What for | Source | Licence |
|---|---|---|---|
| `ggml-small.bin` | live captions | huggingface.co/ggerganov/whisper.cpp | MIT |
| `ggml-large-v3-turbo.bin` | final transcript | huggingface.co/ggerganov/whisper.cpp | MIT |
| `qwen3.5-4b-Q4_K_M.gguf` | summary | built from Qwen/Qwen3.5-4B at `851bf6e`, see `bench/llm_models.json` (bytes, sha256, llama.cpp commit) | Apache-2.0 |
| `silero_vad.onnx` | speech spans | github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/silero_vad.onnx (k2-fsa's export of snakers4/silero-vad v4), 643,854 bytes, sha256 `9e2449e1087496d8d4caba907f23e0bd3f78d91fa552479bb9c23ac09cbb1fd6` | MIT |

The VAD is that onnx file on tract, not whisper.cpp's `ggml-silero-v5.1.2.bin`: the ggml
port reflect-pads each frame instead of carrying the previous frame's 64 samples, and
on FLEURS it lost whole utterances (`bench/README.md`).

Data lives in the app data folder
(`~/Library/Application Support/dev.sshlab.meetbutler/` on macOS): `meetbutler.db` and
`sessions/<id>/{mic,system}.pcm`.

## Manual checks

There is no automated runtime coverage of capture or the window. Run these by hand on a
real Mac before calling a build good, and note the macOS version.

Permissions
- [ ] First recording with both sources: macOS asks for the microphone, then for screen
      recording (system audio comes through ScreenCaptureKit).
- [ ] Deny screen recording: recording still starts from the mic alone, and the window
      says system audio was left out. The session lists `mic` only.
- [ ] Deny the microphone, allow screen recording: recording runs on system audio alone.
- [ ] Deny both: nothing starts, and the window says why.

Recording
- [ ] Nothing records until Start is pressed. While recording, the red indicator and the
      elapsed time are always visible.
- [ ] Play a video in a browser during a call-like test: its speech shows as "상대", your
      voice as "나". MeetButler's own sound is not captured.
- [ ] Captions appear within a few seconds of speech (target: 3 s; measure and record).
- [ ] Stop: the final pass runs ("최종본을 만드는 중… n%"), then the transcript switches
      from grey provisional captions to the final one.

Summary
- [ ] After the final pass, the summary runs on its own ("요약하는 중… n%"), then shows
      the summary, decisions, action items and open questions. Note the time it took and
      the meeting's length.
- [ ] Activity Monitor while it runs: one `meetbutler-summarizer` process; it is gone
      once the summary shows, and the app's memory is back where it was.
- [ ] 취소 during the summary: it stops within a few seconds, shows 취소됨 and
      다시 요약; 다시 요약 runs it again.
- [ ] Without the summary model in the models folder: the transcript is unaffected and
      the summary says it failed, naming the missing file.
- [ ] Cmd+Q during the summary: the app quits promptly and no `meetbutler-summarizer`
      is left running.

Vocabulary (STT-7) — real speech; the benchmark only had synthetic voices
- [ ] Put the attendees' names and a few terms in the vocabulary, hold a short meeting
      that says each one: the final transcript spells them as written. Note any name heard
      as another attendee's name (the benchmark saw this: "시온" heard as "지호").
- [ ] Short replies and pauses: no segment is just vocabulary names repeated
      ("지호, 지호, 지호."); a real repeated reply ("네, 네.") is still there.
- [ ] A term in the vocabulary that nobody says does not appear in the transcript.
- [ ] An empty vocabulary: the final transcript is as before.

Quitting — every path must end without a crash report and keep the recording
- [ ] Cmd+Q while recording: the app quits; on relaunch the session is there, stopped,
      with its audio and captions so far.
- [ ] Close the last window while recording: same as Cmd+Q.
- [ ] Cmd+Q during the final pass: the app quits promptly; the session keeps its live
      transcript (the final pass did not half-apply).
- [ ] Log out (and shut down) while recording: no "quit unexpectedly" dialog on the next
      login; the session is there.
- [ ] Force quit (Activity Monitor / `kill -9`) while recording: on relaunch the session
      shows as interrupted, with audio and captions up to the last pass.

The quit paths matter because a Metal model still loaded when the process exits makes
ggml abort; `AppState::shutdown` stops recording, cancels the final pass and drops every
model on `RunEvent::Exit`.

## Backlog (0.2)

- Finish what a hard exit left behind. A session killed outright (`kill`, a crash, power
  loss) never reaches `RunEvent::Exit`; on the next launch it is `interrupted` with its
  audio and live captions on disk. Run `Recorder::finalize` on such sessions at startup,
  or offer a "최종본 만들기" button in the library; it already works on interrupted
  sessions.
- Windows system audio (WASAPI loopback, `crates/meetbutler-audio/src/system.rs`).

