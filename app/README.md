# MeetButler app (0.2 skeleton)

Tauri 2 shell over `crates/meetbutler-engine` and `crates/meetbutler-audio`. Opens no
port; the windows talk to Rust over Tauri IPC only.

Three windows (requirements §6), all the app's own pages under one capability:

| Window | Page | What |
|---|---|---|
| mini bar (`main`, always on top) | `index.html`, `main.js` | recording state, elapsed time, sources, the newest caption, Start/Stop, the way to the others; closing it quits |
| panel (`panel`, 360×560) | `panel.html`, `panel.js` | 자막 (Captions) / 요약 (Summary) / 기록 (Record) tabs for the meeting on screen; 기록 holds the next meeting's sources, language and vocabulary, and this one's title and Markdown |
| library (`library`) | `library.html`, `library.js` | past meetings with a title filter, summary beside transcript, rename, Markdown copy/save, delete |

`common.js` is shared. Closing the panel or the library hides it.

```bash
# models are read from MEETBUTLER_MODELS, else <app data>/models; the app never downloads them
cargo build --release -p meetbutler-summarizer --features llama   # the summary sidecar, with llama.cpp
MEETBUTLER_MODELS=$PWD/bench/data/models cargo run -p meetbutler --release
cargo test -p meetbutler                      # security posture (no port, CSP, no network crates)
app/bundle.sh                                 # the .app/.dmg with the sidecar inside (needs `cargo tauri`)
```

The bundle needs the Tauri CLI, a development tool only: `cargo install tauri-cli
--version 2.12.3 --locked` (the version this was built with). Built on an M4 with the
release profile's `strip`: the .app is 38 MB (app 30 MB, sidecar 10 MB; models are not in
it), the dmg 13 MB. The whole bundle is signed ad hoc (`signingIdentity: "-"`) with the
hardened runtime and one entitlement, the microphone (`Entitlements.plist`); a copy
downloaded from elsewhere is still stopped by Gatekeeper until it is signed with a
Developer ID and notarized.

`MEETBUTLER_DATA_DIR` points the app at another data folder, so a development run, a
manual check or a screenshot never touches real meetings.

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

There is no automated runtime coverage of capture or the windows. Run these by hand on a
real Mac before calling a build good, in this order (one sitting, about an hour), and
note the macOS version and the Mac. Tick a box only when it held; write what happened
next to one that did not.

**0. Setup — keep real meetings out of it**
- [ ] `app/bundle.sh`, then copy `MeetButler.app` out of the dmg into /Applications.
- [ ] A test folder with the models: `mkdir -p ~/MeetButler-test/models` and copy (or
      link) the four model files there (`ggml-small.bin`, `ggml-large-v3-turbo.bin`,
      `silero_vad.onnx`, `qwen3.5-4b-Q4_K_M.gguf`).
- [ ] Launch it on that folder from Terminal:
      `open -n --env MEETBUTLER_DATA_DIR=$HOME/MeetButler-test --env MEETBUTLER_MODELS=$HOME/MeetButler-test/models /Applications/MeetButler.app`

**1. First launch, icon, menu bar**
- [ ] It opens (it is signed ad hoc and not quarantined; a downloaded copy would not).
- [ ] Dock icon: the whole mascot on a navy tile. Finder list view and the Cmd+Tab
      switcher show the closer crop (face and tray). Both read in light and dark mode.
- [ ] Only the mini bar shows, on top of other apps, draggable by its title bar.
- [ ] Menu bar: the camera-and-voice glyph, tinted to match the menu bar in light and dark;
      its menu has 녹음 시작 (Start recording), 패널 열기 (Open panel), 라이브러리 (Library),
      MeetButler 종료 (Quit MeetButler).
- [ ] ▢ opens the panel, ☰ the library; closing either hides it, and the button or the
      menu brings it back.
- [ ] Activity Monitor: no `meetbutler-summarizer` running.

**2. Permissions** (reset between runs: `tccutil reset Microphone` / `ScreenCapture`)
- [ ] First recording with both sources: macOS asks for the microphone, then for screen
      recording (system audio comes through ScreenCaptureKit). In the bundle this needs
      the microphone entitlement of its hardened runtime.
- [ ] Deny screen recording: recording still starts from the mic alone, and the window
      says system audio was left out. The session lists `mic` only.
- [ ] Deny the microphone, allow screen recording: recording runs on system audio alone.
- [ ] Deny both: nothing starts, and the window says why.

**3. Recording and the windows**
- [ ] Panel 기록 (Record) tab: set sources, language and a vocabulary (the attendees' names and a
      few terms) for the next meeting.
- [ ] Start from the menu bar with the panel closed: it records with that setup. The menu
      bar glyph gets its dot, its tooltip says 녹음 중 (Recording), the first menu item reads 녹음 정지 (Stop
      recording).
- [ ] Nothing records until Start. While recording, the red indicator and the elapsed time
      are always visible in the mini bar; the panel's elapsed time matches it.
- [ ] Play a video in a browser during a call-like test: its speech shows as "상대" (them),
      your voice as "나" (me). MeetButler's own sound is not captured.
- [ ] Captions appear within a few seconds of speech (target: 3 s; measure and record).
      The mini bar shows the newest line.
- [ ] The vocabulary field cannot be changed while recording.
- [ ] Plug a headset in mid-meeting, speak, then pull it out and speak again: "나" keeps
      captioning from whichever microphone is the default; the log shows "mic: the stream
      ended or its device changed; building it again" each time, and `mic.pcm` stays as
      long as the meeting.
- [ ] Stop from the mini bar (another time: start from the panel, stop from the menu bar):
      every window follows.
- [ ] Silence on the other side: with nothing playing on the Mac, record one minute
      (speak now and then), stop, and compare the two recordings in
      `~/MeetButler-test/sessions/<id>/`: `ls -l` the `mic.pcm` and `system.pcm` sizes
      (32,000 bytes a second each). They should match to within a fraction of a second.
      Note the two sizes either way: it tells whether ScreenCaptureKit sends silence or
      nothing while nothing plays.
- [ ] Accent: buttons, the selected tab and the ↗ links are periwinkle, never red or the
      grey of provisional captions; text on them reads in light and dark.

**4. Vocabulary (STT-7)** — real speech; the benchmark only had synthetic voices
- [ ] In the meeting of step 3, say each vocabulary term at least once, plus a few short
      replies ("네, 네." "yes, yes", "맞아요" "right").
- [ ] The final transcript spells the terms as written. Note any name heard as another
      attendee's name (the benchmark saw "시온" heard as "지호").
- [ ] No segment is just vocabulary names repeated ("지호, 지호, 지호."); the real repeated
      replies are still there.
- [ ] A term in the vocabulary that nobody says does not appear in the transcript.
- [ ] Next launch: the field holds this meeting's vocabulary. A meeting with an empty
      vocabulary transcribes as before.

**5. After Stop: final transcript, then summary**
- [ ] The final pass runs ("최종본을 만드는 중… n%", making the final transcript), then the transcript switches from
      grey provisional captions to the final one.
- [ ] The summary follows on its own ("요약하는 중… n%", summarising), then the 요약 (Summary) tab shows the
      summary, decisions, action items and open questions. Note how long it took and the
      meeting's length.
- [ ] Activity Monitor while it runs: one `meetbutler-summarizer` (inside the .app); it is
      gone once the summary shows, and the app's memory is back where it was.
- [ ] A summary item's ↗ switches to 자막 (Captions) and highlights the line it came from; the line
      says what the item says.
- [ ] Another meeting: 취소 (Cancel) during the summary stops it within a few seconds, shows 취소됨
      (Cancelled) and 다시 요약 (Summarise again); 다시 요약 runs it again.
- [ ] Rename `qwen3.5-4b-Q4_K_M.gguf` away and summarise again: the transcript is
      unaffected and the summary says it failed, naming the missing file. Rename it back.

**6. Library and Markdown**
- [ ] Before the first meeting (an empty data folder), the library shows the mascot and
      "아직 회의가 없습니다…" (no meetings yet), and the panel's 요약 tab the mascot and
      "회의가 끝나면…" (when the meeting ends);
      both read in light and dark mode, with no light rim around the figure.
- [ ] The library lists the meetings; the filter narrows by title.
- [ ] Rename in the library: the panel shows the new title, and the other way round.
- [ ] Markdown 복사 (Copy Markdown) pastes into a text editor as the meeting (title, date, summary with
      times, transcript).
- [ ] Markdown 저장 (Save Markdown) writes into Downloads and shows it in Finder; a second save does not
      overwrite the first.
- [ ] Delete: the first click asks again, the second deletes, and the meeting's folder
      under `~/MeetButler-test/sessions/` is gone. A meeting recording or being finalised
      refuses.
- [ ] Dark mode: every window is readable.

**7. Quitting** — every path must end without a crash report and keep the recording
- [ ] Cmd+Q while recording: the app quits; on relaunch the session is there, stopped,
      with its audio and captions so far.
- [ ] Close the mini bar while recording: same as Cmd+Q. MeetButler 종료 (Quit MeetButler) in
      the menu bar: same.
- [ ] Cmd+Q during the final pass: the app quits promptly; the session keeps its live
      transcript (the final pass did not half-apply).
- [ ] Cmd+Q during the summary: the app quits promptly and no `meetbutler-summarizer` is
      left running.
- [ ] Log out (and shut down) while recording: no "quit unexpectedly" dialog on the next
      login; the session is there.
- [ ] Force quit (Activity Monitor / `kill -9`) while recording: on relaunch the session
      shows as interrupted, with audio and captions up to the last pass.

**8. Windows** (when a Windows machine is at hand; system audio there has only been
built, never run)
- [ ] A call with a headset: "나" and "상대" both caption, and nothing echoes.
- [ ] A quiet minute with nothing playing, then speech: `mic.pcm` and `system.pcm` have
      the same size to within a fraction of a second, and later captions are not early.
      The log says how much silence was filled ("system: … filled with silence").
- [ ] Plug a headset in mid-meeting (and pull it out): recording carries on; the log
      shows "system: the stream ended; building it again"; the hole is silence.
- [ ] Speakers, no headset: note how much of "상대" also shows as "나".
- [ ] Settings > Privacy > Microphone off for desktop apps: recording runs on system
      audio alone, and the window says the microphone was left out.
- [ ] Silent packets: with nothing playing, `system.pcm` holds zeros, not noise (open it
      as raw 16 kHz s16le in Audacity, or check its peak).

The quit paths matter because a Metal model still loaded when the process exits makes
ggml abort; `AppState::shutdown` stops recording, cancels the final pass and the summary,
and drops every model on `RunEvent::Exit`.

## Backlog (0.2)

- Finish what a hard exit left behind. A session killed outright (`kill`, a crash, power
  loss) never reaches `RunEvent::Exit`; on the next launch it is `interrupted` with its
  audio and live captions on disk. Run `Recorder::finalize` on such sessions at startup,
  or offer a "최종본 만들기" (Make the final transcript) button in the library; it already works on interrupted
  sessions.
- Windows system audio (WASAPI loopback, `crates/meetbutler-audio/src/system.rs`).

