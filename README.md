# ghostmeet

Self-hosted AI meeting delegate — live captions and summaries from any browser tab.

## Current status
- ✅ Step 1: Architecture/scope locked
- ✅ Step 2: Chrome tab audio capture → WebSocket → local backend
- ✅ Step 3: Real-time STT pipeline (faster-whisper)
- ✅ Step 4: Extension live captions UI (side panel)
- ⏳ Step 5+: Summary engine, agent mode pending

See `IMPLEMENTATION_PLAN.md` for full roadmap.

---

## Quick Start

### 1) Start backend
```bash
cd ghostmeet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

# default model: base (options: tiny, base, small)
GHOSTMEET_MODEL=base python -m backend
```

### 2) Load Chrome extension
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked** → select `extension/` folder

### 3) Use it
1. Open a meeting tab (Google Meet, Zoom web, etc.)
2. Click ghostmeet extension icon → **Start Capture**
3. Side panel opens automatically with live captions
4. Click **Stop Capture** when done
5. View transcript: `GET http://127.0.0.1:8877/api/sessions/{id}/transcript`

---

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/api/health` | GET | Health check + model info |
| `/api/sessions` | GET | List all sessions |
| `/api/sessions/{id}` | GET | Session details |
| `/api/sessions/{id}/transcript` | GET | Full transcript |
| `/ws/audio` | WS | Audio ingest |
| `/ws/transcript/{id}` | WS | Live transcript stream |

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `GHOSTMEET_MODEL` | `base` | Whisper model size (`tiny`/`base`/`small`) |
| `GHOSTMEET_DEVICE` | `auto` | Compute device (`auto`/`cpu`/`cuda`) |
| `GHOSTMEET_LANGUAGE` | auto-detect | Language code (`en`/`ko`/`ja`/etc.) |

---

## Repo Layout

```text
ghostmeet/
├── extension/              # Chrome MV3 extension
│   ├── manifest.json
│   ├── background.js       # tab audio capture + WebSocket
│   ├── popup.html/js       # start/stop controls
│   ├── sidepanel.html      # live captions UI
│   ├── sidepanel.js        # transcript WebSocket client
│   └── sidepanel.css       # dark theme styles
├── backend/                # FastAPI backend
│   ├── app.py              # main server + routes
│   ├── audio_processor.py  # ffmpeg webm→PCM conversion
│   ├── transcriber.py      # faster-whisper wrapper
│   └── models.py           # data models
├── recordings/             # captured audio (runtime)
├── IMPLEMENTATION_PLAN.md
├── requirements.md
└── requirements.txt
```
