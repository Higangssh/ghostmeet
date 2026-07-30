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

**Before every commit:** run `git status` and `git diff --staged` and read them for
secrets and recordings. `.gitignore` is a safety net, not a substitute for looking.

**Test fixtures must be synthetic.** Generate audio programmatically (see
`make_webm_opus` in `tests/test_decoder.py` — a sine wave through a real opus encoder).
Never check in a clip of an actual meeting.

**Do not weaken the local-only posture** — it is the project's core promise:
- Bind `127.0.0.1` by default, never `0.0.0.0`
- Do not widen CORS to `*`; scope it to the extension origin
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
- **Absolute timestamps** are derived from the sample cursor, not from per-chunk
  accumulation — speaker diarization will depend on them lining up.
- `Segment.speaker` exists and is `None` until diarization lands. Keep it in the wire
  format.

## Dev

```bash
python -m venv .venv && ./.venv/Scripts/python.exe -m pip install -r requirements.txt
./.venv/Scripts/python.exe -m pytest tests/ -q     # tests must not download a Whisper model
./.venv/Scripts/python.exe -m backend             # run backend on :8877
```

Tests inject fake transcribe functions and duck-typed audio sources so the suite stays
fast and offline. Keep it that way — no test should need a model or a network call.

## Testing discipline

TDD: write the failing test, watch it fail for the right reason, then implement. When a
test catches a bug, verify the test actually fails without the fix before moving on.
