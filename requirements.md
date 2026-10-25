# MeetButler — requirements v2

> Formerly ghostmeet. v0.1.0 (Chrome extension + local backend) is kept as a tag; this
> document starts the desktop app line. The previous version: `git show v0.1.0:requirements.md`.

## 1. In one line

**Wherever the meeting happens — a meeting room or Meet — my computer listens, sorts it out, and remembers.**

A "super meeting app" does not mean a transcriber. One app takes care of the meeting
**before, during and after**:

| Stage | What the butler does |
|---|---|
| Before | Recognises the meeting from the calendar and briefs on what is still open from past meetings |
| During | Records, shows live captions, tells who spoke, and takes marks on important moments |
| After | Pulls out the summary, decisions and action items, exports them, and brings every meeting back through search and questions |

## 2. Principles — before features

1. **Everything runs on local models.** The goal is that transcription, diarization,
   summary, questions and search all work end to end with models on the device. Audio and
   transcripts never leave the device. A cloud LLM (Claude etc.) is only an **option** the
   user picks, never the default.
2. **Recording is visible.** While recording, an indicator is always on screen, and nothing
   starts before the user presses it. We do not build a feature that hides "recording" in
   an in-person meeting.
3. **Long meetings are the default.** At 4 hours and more, one update still costs the same.
   A change that reprocesses the whole session is a regression.
4. **Stay out of the way during the meeting.** The default screen is a one-line mini bar.
5. **No telemetry, analytics or crash reports.** No external SaaS dependency.

## 3. Where it is used

| Scene | Audio source | Clue to who is speaking |
|---|---|---|
| In-person meeting in a room (laptop on the table) | Microphone | Voice only — diarization required |
| Web meeting (Meet, Zoom web, Teams web) | Microphone + system audio | Channel (me / them), speaker names from the extension |
| Desktop meeting app (Zoom, Teams app) | Microphone + system audio | Channel (me / them) |
| Past recordings (phone recording, recorder file) | File import | Voice only |

## 4. Platforms

- **Windows 10/11, macOS 13+ (Apple Silicon first, Intel best effort)**
- Desktop app: Tauri 2 (Rust + system WebView)
- System audio: Windows WASAPI loopback, macOS ScreenCaptureKit (13+) / Core Audio
  process tap (14.4+). No virtual audio device (BlackHole etc.) is required.
- Chrome extension: an **optional** companion. The app works fully without it. It talks to
  the app only through Chrome Native Messaging — the app **opens no port.** The extension
  sends no audio, only meeting information (title, participants, speaker).
- Linux, mobile: out of scope (revisit after v1)

## 5. Features

Priority: **P0** cannot ship without it · **P1** the goal of that version · **P2** nice to have.
Versions follow the roadmap in §8.

### 5.1 Recording

| ID | Feature | Priority | Version |
|---|---|---|---|
| REC-1 | Record the microphone, system audio, or both | P0 | 0.2 |
| REC-2 | Store the two sources **per channel** (me = microphone, them = system) | P0 | 0.2 |
| REC-3 | Start and stop from a global shortcut and the tray | P0 | 0.2 |
| REC-4 | Remove the far side's voice that leaks back into the microphone when speakers are used. After the meeting, software echo cancellation (sonora 0.2.0, AEC3, BSD-3-Clause) runs on the two channels before the final pass, with the reference (system) moved 20 ms earlier (a setting added post hoc to remove the zero-delay leak). SPK-2 relies on this: on synthetic calls a -25 dB leak alone takes call DER from 0.101→0.44, and sonora+20 ms brings it to 0.103~0.105 (across delay, room and drift, `docs/rec4-aec-research.md`). Fallbacks: macOS VoiceProcessingIO, Windows communications-mode AEC | P0 | 0.3 |
| REC-5 | "Important" mark during the meeting (one shortcut) | P1 | 0.3 |
| REC-6 | Import audio files (m4a, mp3, wav, webm) | P1 | 0.3 |
| REC-7 | Recording carries on when an input device drops or changes | P1 | 0.3 |
| REC-8 | Recording notice template (one line to read at the start of an in-person meeting) | P2 | 0.4 |

### 5.2 Transcription

| ID | Feature | Priority | Version |
|---|---|---|---|
| STT-1 | Local live transcription, language choice (auto / Korean / English …) | P0 | 0.2 |
| STT-2 | Flat update cost in sessions of 4 hours and more (only the tail window is processed) | P0 | 0.2 |
| STT-3 | Saved as recognised — a crash loses only the current window | P0 | 0.2 |
| STT-4 | GPU acceleration: macOS Metal/CoreML, Windows Vulkan (CUDA optional) | P1 | 0.2 |
| STT-5 | Model choice and download management (a recommendation fitted to the device on first run) | P1 | 0.2 |
| STT-6 | After the meeting, transcribe again with a larger model and replace with the final pass (two passes, §5.2.1) | P0 | 0.2 |
| STT-7 | User vocabulary (participant names and terms given to transcription as a hint) | P1 | 0.2 |

#### 5.2.1 Engine (settled 2026-10-19, evidence: `bench/results/RESULTS.md`)

- **whisper.cpp only**, called from Rust through whisper-rs (Unlicense; the whisper.cpp
  inside is MIT). The whole app is Rust.
- **Two passes.** During the meeting a small model shows **provisional captions** (measured
  CER about 8%); when the meeting ends a turbo model transcribes again from the start and
  replaces them at once with the **final pass** (measured 2.7% on the Rust path). Screens and
  storage tell provisional from final.
- **We cut the speech spans.** Silero VAD (MIT, v4 onnx) runs on tract, a pure Rust ONNX
  runtime, to cut spans, and each span is decoded. whisper.cpp's built-in Silero v5 was
  dropped because it missed speech in real voices (FLEURS CER 6.0%,
  `bench/results/RESULTS.md` section 7). Live and after the meeting use **the same function
  and the same values**. We do not leave it to an engine's built-in VAD.
- **Hallucination blocklist.** A span under 2 s that matches a known hallucinated phrase
  ("시청해주셔서…" ("thanks for watching…") etc.) is dropped. Regression tests guard that a
  real short reply ("네" ("yes"), "감사합니다" ("thank you")) is not removed.
- Pass bar: live CER 10% or less, final 3% or less (FLEURS 36 min). Short replies are judged
  by recall — at least 51 of 53, none lost to the blocklist. The final transcript of a 4-hour
  meeting takes about 23 min on an M4 (RTF 0.094).
- When changing the VAD or the engine, **do not judge on synthetic speech alone.** v5 looked
  better on synthetic TTS and was worse on real voices.
- Excluded: faster-whisper (Python), SenseVoice (model licence limited to "reference and
  study"). SenseVoice stays only as a candidate optional engine the user downloads.

### 5.3 Speakers

| ID | Feature | Priority | Version |
|---|---|---|---|
| SPK-1 | Channel-based split: me / them (web and app meetings) | P0 | 0.2 |
| SPK-2 | Voice-based diarization (in-person meetings, several people in the far-side channel). After the meeting, piece-wise clustering + global link, ERes2NetV2 embeddings (`bench/results/RESULTS.md` §9). Speaker cap 20, 1 GB memory target for 4 hours | P1 | 0.3 |
| SPK-3 | Name the speakers — one correction applies to the whole meeting | P1 | 0.3 |
| SPK-4 | Link names automatically from the Meet speaker info the extension gives | P2 | 0.4 |
| SPK-5 | Recognise the same voice across meetings (local voiceprint, only for people who agree) | P2 | 1.x |

### 5.4 Summary

| ID | Feature | Priority | Version |
|---|---|---|---|
| SUM-1 | Summary · decisions · action items (owner, due date) — a fixed output format | P0 | 0.2 |
| SUM-2 | Summary with a local LLM (default). A cloud LLM (Claude, user's key) is optional | P0 | 0.2 |
| SUM-3 | Jump from each summary item to the line it rests on (timestamp link) | P0 | 0.2 |
| SUM-4 | Templates per meeting type (1:1, weekly, interview, customer meeting) | P1 | 0.4 |
| SUM-5 | Pre-meeting briefing input (topic, my role, what to say) reflected in the summary | P2 | 0.4 |

The default summary model is **Qwen3.5-4B (Apache-2.0), summarising the meeting in parts and
then merging** (2026-10-20, `bench/results/RESULTS.md` section 8). It runs in 3.2GB even on
8GB devices and summarises a 1-hour meeting in about 100 s on an M4. 9B has the same recall
and is twice as slow, so it stays only as an option for 16GB devices. A summary can write a
wrong amount or date, and checking it against the transcript does not catch that (the number
is in the transcript), so **0.2 ships a link from every decision to its evidence (SUM-3).** On
the synthetic speech route the largest loss was not the summary but transcribing names and
proper nouns, so the vocabulary (STT-7) is pulled into 0.2 as well.

The local LLM runs as a **separate executable shipped with the app (sidecar)** (decided
2026-10-19). It is a small Rust executable that links only llama.cpp, talks to the app only
over stdin/stdout, and opens no port. Putting it in one binary with whisper.cpp made two ggml
versions clash and killed the llama side. Kept apart, memory surely comes back when the
summary ends, and the two engines can be upgraded separately. Connecting to Ollama is not done:
it needs a separate install and an HTTP client in the app.

### 5.5 Memory

| ID | Feature | Priority | Version |
|---|---|---|---|
| MEM-1 | Meeting list, per-meeting transcript and summary view, rename, delete | P0 | 0.2 |
| MEM-2 | Full-text search across all meetings | P1 | 0.3 |
| MEM-3 | Questions over all meetings ("What did we conclude about the budget last month?") | P1 | 0.4 |
| MEM-4 | **MCP server** — Claude Desktop, Claude Code etc. look up the meeting records | P1 | 0.4 |
| MEM-5 | Open action items in one view (across meetings) | P2 | 0.5 |

### 5.6 Connections

| ID | Feature | Priority | Version |
|---|---|---|---|
| OUT-1 | Markdown export / copy | P0 | 0.2 |
| OUT-2 | Save to an Obsidian folder automatically | P2 | 0.4 |
| CAL-1 | Calendar link: fill in meeting title and participants, start reminder | P2 | 0.5 |
| EXT-1 | Chrome companion extension: meeting title, participants, speaker (Native Messaging) | P2 | 0.4 |

### 5.7 Agent Mode

v0.1's planned Agent Mode (speaking on your behalf) is **postponed until after v1.** In-person
meeting support and trust (§2, item 2) come first. When it comes back, a separate document
sets its policy and guardrails first.

## 6. Screens

- **Mini bar** (default, always on top, drag to pin its place): recording indicator · elapsed
  time · sources · last caption line · important mark · expand
- **Panel** (about 360×560): captions / summary / notes tabs, with source status and stop below
- **Library window**: past meetings, search, questions — the screen used outside meetings
- Start from anywhere with the tray (menu bar) icon
- Light/dark theme, Korean and English UI
- Brand: follows the butler series rules (on the mascot's tray = meeting objects). Red is
  used for the recording indicator, so it is left out of the brand colours.

## 7. Non-functional targets

The figures are **targets**, to be corrected with measurements once the first task of §8
(the engine comparison) is done.

| Item | Target |
|---|---|
| Caption delay | On screen within 3 s of speaking (M1 / recent 6-core Windows, small-class model) |
| Long sessions | In a 4-hour session, update time and memory at the same level as the first 10 minutes |
| Memory | 1.5GB or less while recording and transcribing (models included) |
| Installer | 50MB or less without models |
| Crash | On restart, recording and transcript remain up to just before the last window |
| Data location | One folder the user can choose (easy to back up and delete) |

## 8. Roadmap

| Version | Goal | Core |
|---|---|---|
| 0.1.0 | (archived) ghostmeet extension + local backend | Tagged |
| **0.2** | **First desktop version** — record anywhere, transcribe locally, and summarise | REC-1~3, STT-1~7, SPK-1, SUM-1~3, MEM-1, OUT-1 |
| 0.3 | In-person meetings done properly | Diarization, echo cancellation, important mark, file import, search |
| 0.4 | A butler that remembers | Questions over all meetings, MCP, templates, companion extension |
| 0.5 | A butler that links before and after | Calendar, open action items view |
| 1.0 | Stable | Signing and auto-update, install within 10 minutes |

**The engine comparison, 0.2's first task, is done** (§5.2.1, `bench/`). No real meeting
recordings were used in the measurements. 0.2's milestones:

1. **App skeleton and engine** — the Tauri app takes the microphone and system audio
   separately, and a Rust engine crate does Silero span cutting + whisper-rs transcription +
   the hallucination blocklist. Python tests that hold the app's promises (4-hour structure,
   absolute timestamps, storage on disk, SQLite, keeping the session on failure) move into
   Rust specs.
2. Live captions → replaced by the final pass after the meeting (the whole session replaced in
   one transaction)
3. **Local summary model measurement** — look at Korean summary quality and speed on the same
   public transcripts.
4. Summary, library, Markdown export

Since everything runs locally, several models live on one device at once (transcription +
speakers + LLM + embeddings). The default layout loads **only transcription and speakers
during the meeting, the LLM after it**, to keep the memory target (§7).

## 9. Versioning

- SemVer. Before 1.0, a minor is a feature batch and a patch is a fix
- Keep Conventional Commits (`feat:`, `fix:`, `docs:` …)
- Update `CHANGELOG.md` every version and release with a `vX.Y.Z` tag
- The pre-release privacy scan (CLAUDE.md) also applies to tag pushes

## 10. Renaming

- Repository `ghostmeet` → `meetbutler` (keep GitHub's redirect — do not create a `ghostmeet`
  repository again on the same account)
- Site `ghostmeet.sshlab.dev` → `meetbutler.sshlab.dev`, old address redirected with 301
- Publish the ClawHub skill under the new name, with a move notice on the old name
- Environment variables use the new names but still read `GHOSTMEET_*` for a while
- Keep the "formerly ghostmeet" note on the README's first line for two versions

## 11. Not allowed

- External SaaS dependencies (Recall.ai, Deepgram etc.)
- Starting a recording the user did not press, hiding the recording indicator
- Sending audio or transcripts off the device (except an LLM call the user asked for)
- Hard-coded API keys, test material or screenshots holding real meeting content

## 12. Risks

| Risk | Impact | Response |
|---|---|---|
| Quality of live Korean diarization (Sortformer is mostly 4 speakers, English) | High | Measured (`bench/results/RESULTS.md` §9): live changes over 40% of labels, so **diarization after the meeting is the default**. Settled on piece-wise clustering + global link: ERes2NetV2 ranks first with in-person DER 0.169 (CAM++ 0.225), vanished speakers 15→1. Remaining issues: about 1.05 GB of memory for 4 hours on the bench (target 1 GB), 25~37 min of processing, speaker over-counting (2 people as 7) — handled with the speaker cap and SPK-3 merging |
| macOS system audio permission flow | Medium | First-run guide screen; without permission, works with the microphone only |
| Windows GPU variety | Medium | Vulkan by default, automatic fallback to CPU on failure |
| Cost of the Rust port | Medium | Port settled. The Python design (PcmStore, tail window) and tests move into specs |
| Code signing and notarisation cost | Low | Unsigned for 0.x, solved before 1.0 |
| Meet DOM changes | Low | The extension is optional; the app works if it breaks |

## 13. Open questions

- Memory target for the local LLM: a separate cap for the summary after the meeting (e.g. it
  works on 8GB devices too)
- Licence: keep it as it is (or match the sibling products)
