# ghostmeet — working notes for Claude

Self-hosted meeting transcription: Chrome MV3 extension captures tab audio → local
FastAPI backend (faster-whisper) → live captions + on-demand Claude summary.

## Security — this is a PUBLIC repo handling private meeting audio

Treat every change as world-readable and every recording as personal data.

**Never commit:**
- `.env`, API keys, tokens, or anything resembling `sk-ant-…` (only `.env.example` with commented-out placeholders)
- `recordings/`, `*.webm`, `*.pcm`, `*.wav` — these are real meeting audio
- Real transcript or summary text, in code, tests, docs, commit messages, or issue/PR bodies
- `demo/` contents, screenshots, or GIFs that show real participants, names, or discussion

**Before every push**, not just every commit, run the scan below and report what it
found. `.gitignore` is a safety net, not a substitute for looking.

```bash
git status && git diff --staged                       # read these, do not skim
git grep -nIiE "C:\\\\Users|/home/[a-z]|AppData|[a-z0-9._%+-]+@[a-z0-9.-]+\.[a-z]{2,}"
git grep -nIiE "sk-ant-|ghp_|AKIA[0-9A-Z]{16}|-----BEGIN.*PRIVATE KEY"
git log --all --pretty=format: --name-only --diff-filter=A | sort -u \
  | grep -iE "\.env$|recordings/|\.webm$|\.pcm$|\.wav$|\.db$"   # history, not just HEAD
```

Expected hits, safe to ignore: the pattern lines in this file, and
`.env.example:2` (`sk-ant-...`, a placeholder). Anything else is a finding.

**Grep does not read images.** Screenshots, GIFs and the demo asset can show real
meeting content, names or faces that no text scan will catch. Look at them. Frames come
out with `av.open(path)` plus `frame.to_image().save(...)`. `assets/demo.gif` was checked
on 2026-08-02 and is a synthetic meeting (Alice Chen / Bob Smith / Charlie Park, emoji
avatars) — re-check if it is ever replaced.

The commit author email is public in every commit going back to the first one, and the
repo owner has decided that is fine. Do not raise it again.

**Test fixtures must be synthetic.** Generate audio programmatically (see
`make_webm_opus` in `tests/test_decoder.py` — a sine wave through a real opus encoder).
Never check in a clip of an actual meeting.

**Do not weaken the local-only posture** — it is the project's core promise:
- Bind `127.0.0.1` by default. The Docker image sets `0.0.0.0` because it has to listen
  inside the container; compose publishes the port on the host's loopback only. Keep
  both halves of that arrangement intact.
- Do not widen CORS to `*`; it is scoped to extension and loopback origins
- No telemetry, no analytics, no crash reporting
- The only permitted outbound call is the user-triggered Anthropic summarize request
- Do not introduce third-party SaaS dependencies (Recall.ai, Deepgram, etc.)

**Commit messages** describe the change, not the meeting. No customer names, no
internal project names, no pasted transcript excerpts.

## Design constraints

- **Long sessions are a requirement, not a nice-to-have.** Users asked for 4+ hour
  meetings. Per-pass transcription cost must stay flat: audio lives on disk
  (`PcmStore`), decoding is incremental (`StreamingWebmDecoder`, one demuxer per
  session), and Whisper only ever sees a bounded tail window (`IncrementalTranscriber`).
  Any change that reintroduces "re-process the whole session" is a regression.
- **Load the Whisper model once per process**, never per session.
- **Recording happens in an offscreen document, never in the service worker.**
  `chrome.tabCapture.capture()` is documented foreground-only and `MediaRecorder` does
  not exist in a service worker scope. The worker calls `getMediaStreamId()` and hands
  the id to `offscreen.js`. Capturing a tab also silences it, so the offscreen document
  plays the stream back through an `AudioContext` — do not remove that.
- Summary text is model output derived from meeting audio. Escape it before it reaches
  `innerHTML` (`renderSummary` in `extension/shared.js`).
- **Absolute timestamps** are derived from the sample cursor, not from per-chunk
  accumulation — speaker diarization will depend on them lining up.
- `Segment.speaker` exists and is `None` until diarization lands. Keep it in the wire
  format.
- **Segments are persisted as they are recognised**, not at the end of the meeting, so a
  crash costs at most the current window. `SessionStore` is shared across threads under a
  lock — the transcription worker writes, the event loop reads.
- Live sessions answer `/transcript` from the running pipeline; finished ones read from
  SQLite. Both paths must stay in sync in shape.

## Dev

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt
./.venv/Scripts/python.exe -m pytest tests/ -q          # tests must not download a Whisper model
node --test tests/extension/shared.test.mjs             # extension pure helpers
./.venv/Scripts/python.exe -m backend                   # run backend on :8877
```

The extension has no automated runtime coverage. Verify capture by hand: load
`extension/` unpacked at `chrome://extensions`, open a tab with audio, click Start, and
check `curl http://127.0.0.1:8877/api/sessions`. Driving a real MV3 extension from
Playwright or raw CDP did not work here — the service worker never registered under
`--load-extension`, so those runs prove nothing either way.

Tests inject fake transcribe functions and duck-typed audio sources so the suite stays
fast and offline. Keep it that way — no test should need a model or a network call.

## Testing discipline

TDD: write the failing test, watch it fail for the right reason, then implement. When a
test catches a bug, verify the test actually fails without the fix before moving on.
