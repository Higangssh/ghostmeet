<p align="center">
  <img src="assets/logo.jpg" width="200" alt="ghostmeet logo">
</p>

# ghostmeet

**Your invisible AI meeting assistant** — Live captions and smart summaries, right in your browser.

[Quick Start](#quick-start) • [How It Works](#how-it-works) • [Features](#features) • [API](#api)

---

<p align="center">
  <img src="assets/demo.gif" alt="ghostmeet demo" width="800">
</p>

## What is ghostmeet?

ghostmeet silently captures audio from any browser tab — Google Meet, Zoom, Teams, or anything with sound — and transcribes it in real-time using Whisper. When the meeting ends, click **Summarize** and AI extracts key decisions, action items, and next steps.

It runs as a **Chrome Extension side panel**. Other participants can't see it. Like a ghost in your meeting. 👻

- **100% local** — audio never leaves your machine
- **No accounts** — no sign-up, no cloud, no subscriptions
- **Works everywhere** — any tab that plays audio

## Features

- 🎙️ **Real-time transcription** — Whisper STT, updates every 10 seconds
- ⏱️ **Built for long meetings** — cost per pass stays flat, so a 4-hour session behaves like a 4-minute one
- 📋 **AI-powered summaries** — Key decisions, action items, next steps
- 💾 **Nothing is lost** — transcripts are written to SQLite as they happen and survive a restart
- 🌏 **Per-meeting language** — pick the language in the side panel, or let Whisper detect it
- 🔒 **Self-hosted** — your audio stays on your machine, and the server listens on loopback only
- 🐳 **One-command setup** — `docker compose up` and you're ready
- 👻 **Invisible** — side panel UI, no one in the meeting knows

## How It Works

```
Browser Tab (Zoom / Meet / Teams)
    │ audio
    ▼
Chrome Extension
    ├── service worker  — asks Chrome for a tab stream id
    └── offscreen page  — records it, and plays it back so you still hear the meeting
    │
    ▼  WebSocket (webm/opus, 1s chunks)
Local Backend (FastAPI)
    ├── one demuxer per session  ──→ PCM appended to disk
    ├── Whisper reads only the newest window, never the whole recording
    ├── segments ──→ SQLite  +  live captions in the side panel
    └── Claude API (on demand) ──→ Meeting Summary
```

Everything runs on your machine. The only external call is to Claude API when you click Summarize (optional — transcription works without it).

Audio is decoded once as it arrives and kept on disk, and each transcription pass reads
only a bounded window of it. That is what keeps a long meeting from getting slower and
slower, and keeps memory flat no matter how long you record.

## Quick Start

### Prerequisites

- **Docker** (recommended) or Python 3.10+
- **Chrome** browser

### 1) Start the backend

```bash
git clone https://github.com/Higangssh/ghostmeet.git
cd ghostmeet

# Copy and edit config (add your Anthropic API key for summaries)
cp .env.example .env

# Start with Docker
docker compose up -d
```

Backend is ready when you see `http://0.0.0.0:8877` in the logs.

<details>
<summary><strong>Manual install (without Docker)</strong></summary>

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python -m backend
```

Note: First run downloads the Whisper model (~150MB for `base`).
</details>

### 2) Install Chrome Extension

1. Open **chrome://extensions** in Chrome
2. Enable **Developer mode** (toggle in top-right)
3. Click **Load unpacked**
4. Select the `extension/` folder from this repo
5. Pin the 👻 icon in your toolbar

### 3) Use it

1. **Join a meeting** — Open Google Meet, Zoom, Teams (or any tab with audio)
2. **Click 👻** — Side panel opens on the right
3. **Pick a language** (optional) — or leave it on Auto-detect
4. **Click ▶ Start** — Live captions appear as people speak. The tab stays audible.
5. **Click ■ Stop** — The panel says "Transcription complete" once the last pass finishes
6. **Click 📋 Summarize** — AI generates a structured summary

Reopening the side panel mid-meeting brings the transcript so far back with it.

That's it. No sign-up, no config, no cloud.

## Configuration

Set these in `.env` or `docker-compose.yml`:

| Variable | Default | Description |
|----------|---------|-------------|
| `GHOSTMEET_MODEL` | `base` | Whisper model size (`tiny` / `base` / `small` / `medium` / `large`) |
| `GHOSTMEET_DEVICE` | `auto` | Compute device (`auto` / `cpu` / `cuda`) |
| `GHOSTMEET_COMPUTE_TYPE` | `float32` | Precision (`int8` is much faster on CPU) |
| `GHOSTMEET_LANGUAGE` | auto-detect | Default language (`en` / `ko` / `ja` / etc.) — the side panel can override it per meeting |
| `GHOSTMEET_CHUNK_INTERVAL` | `10` | Seconds between transcription updates |
| `GHOSTMEET_ANTHROPIC_KEY` | — | Required for AI summaries |
| `GHOSTMEET_HOST` | `127.0.0.1` | Server bind address (loopback — the API has no auth) |
| `GHOSTMEET_PORT` | `8877` | Server port |

**Model size guide:**
- `tiny` — fastest, least accurate (~75MB)
- `base` — good balance (recommended, ~150MB)
- `small` — better accuracy, slower (~500MB)
- `medium` / `large` — best accuracy, needs GPU

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check + model info |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}` | GET | Session details |
| `/api/sessions/{id}/transcript` | GET | Full transcript |
| `/api/sessions/{id}/summarize` | POST | Generate AI summary |
| `/api/sessions/{id}/summary` | GET | Get generated summary |
| `/ws/audio` | WS | Audio ingest (binary chunks; send `stop` as text to finish) |
| `/ws/transcript/{id}` | WS | Live transcript stream |

Sessions, transcripts and summaries are stored in `recordings/ghostmeet.db`, so every
endpoint above keeps working after the backend restarts.

## Development

```bash
python -m venv .venv
./.venv/Scripts/python.exe -m pip install -r requirements-dev.txt

./.venv/Scripts/python.exe -m pytest tests/ -q       # backend suite
node --test tests/extension/shared.test.mjs          # extension helpers
```

The suite needs no Whisper model and no network — real opus audio goes through the real
decoder, and only inference is stubbed.

There is one thing tests cannot reach: `chrome.tabCapture.getMediaStreamId()` needs the
`activeTab` grant that only a real toolbar click produces. Everything after that point is
covered by `node tests/extension/verify-capture.mjs`, which drives the extension in a real
browser against a running backend (needs `npm install playwright && npx playwright install
chromium`). To check the last step by hand: start a capture on a tab with audio and confirm
`audio_bytes` climbs in `/api/sessions` — and that you can still hear the tab.

## Project Structure

```
ghostmeet/
├── extension/              # Chrome MV3 Extension
│   ├── manifest.json       # permissions + side panel config
│   ├── background.js       # service worker: gets a tab stream id, drives capture
│   ├── offscreen.html/js   # hidden page that actually records → WebSocket
│   ├── sidepanel.html/js   # live captions, language picker, summaries
│   ├── popup.html/js       # start/stop controls
│   ├── shared.js           # pure helpers shared by the above
│   └── icons/
├── backend/                # Python backend (FastAPI)
│   ├── app.py              # HTTP + WebSocket server
│   ├── decoder.py          # streaming webm/opus → PCM (one demuxer per session)
│   ├── pcm_store.py        # append-only audio on disk, windowed reads
│   ├── incremental.py      # bounded-window transcription, absolute timestamps
│   ├── pipeline.py         # receive / decode / transcribe, decoupled
│   ├── transcriber.py      # shared Whisper model + per-session language
│   ├── store.py            # SQLite: sessions, segments, summaries
│   ├── summarizer.py       # Claude API integration
│   └── models.py           # session model
├── tests/                  # pytest suite + extension tests
├── assets/                 # logo, demo GIF
├── docker-compose.yml      # one-command deployment
├── Dockerfile              # backend container
└── requirements.txt        # Python dependencies
```

## OpenClaw Integration

ghostmeet works as an [OpenClaw](https://github.com/openclaw/openclaw) skill. Control your meetings from chat.

```bash
# Install the skill
clawhub install ghostmeet
```

Then just ask your AI assistant:

- **"Summarize my last meeting"** → generates AI summary from latest session
- **"How many meetings did I have today?"** → lists all sessions
- **"What was discussed?"** → fetches full transcript
- **"Extract action items"** → pulls tasks from the summary

> The skill handles session listing, transcript retrieval, and summary generation via the ghostmeet API. Recording start/stop is done through the Chrome Extension.

## Roadmap

- [x] Real-time transcription (Whisper)
- [x] Chrome Extension side panel UI
- [x] AI meeting summaries (Claude)
- [x] Long meetings — flat cost per pass, tested to 4h+ of audio
- [x] Transcripts survive a restart (SQLite)
- [x] Per-session language selection
- [ ] Meeting context input + file attach
- [ ] Speaker diarization (who said what)
- [ ] In-person meetings (microphone capture)
- [ ] Export to Markdown / PDF
- [ ] Agent Mode — AI speaks in the meeting for you

## License

[MIT](LICENSE)
