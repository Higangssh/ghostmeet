---
name: ghostmeet
description: AI meeting assistant via ghostmeet. Start sessions, get live transcripts, and generate AI summaries from any browser meeting.
metadata:
  {
    "openclaw": {
      "emoji": "👻",
      "requires": { "anyBins": ["docker", "curl"] },
      "envHints": ["GHOSTMEET_ANTHROPIC_KEY"]
    }
  }
---

# Ghostmeet — AI Meeting Assistant

Control [ghostmeet](https://github.com/Higangssh/ghostmeet) from chat. Self-hosted meeting transcription with Whisper + AI summaries.

## Prerequisites

ghostmeet backend must be running (Docker):

```bash
# Quick start
git clone https://github.com/Higangssh/ghostmeet.git
cd ghostmeet
cp .env.example .env
# Edit .env: set GHOSTMEET_ANTHROPIC_KEY for AI summaries
docker compose up -d
```

Chrome Extension must be installed in developer mode from `extension/` folder.

Default backend: `http://127.0.0.1:8877`

## What This Skill Can Do

- **List sessions** → query recorded meeting sessions
- **Fetch transcripts** → retrieve full text from a session
- **Generate summaries** → trigger AI summary via Claude API (requires `GHOSTMEET_ANTHROPIC_KEY`)
- **Health check** → verify backend is running

## What This Skill Cannot Do

- **Start/stop recording** → must be done manually via the Chrome Extension
- **Install the Chrome Extension** → user must load it in developer mode from `extension/` folder
- **Access browser audio** → only the Chrome Extension captures audio; this skill only reads API results

## Required Environment Variables

- `GHOSTMEET_ANTHROPIC_KEY` — required for AI summary generation. Without it, transcription still works but summaries will fail.

## API Commands

### Health Check
```bash
curl -s http://127.0.0.1:8877/api/health
```
Returns: `{"ok": true, "service": "ghostmeet-backend", "model": "base", "chunk_interval_sec": 10}`

### List Sessions
```bash
curl -s http://127.0.0.1:8877/api/sessions
```
Returns list of all meeting sessions with IDs, start times, and segment counts.

### Get Transcript
```bash
curl -s http://127.0.0.1:8877/api/sessions/{session_id}/transcript
```
Returns full transcript with timestamps and text segments.

### Generate Summary
```bash
curl -s -X POST http://127.0.0.1:8877/api/sessions/{session_id}/summarize
```
Triggers AI summary generation (requires `GHOSTMEET_ANTHROPIC_KEY`).
Returns: key decisions, action items, and next steps.

### Get Summary
```bash
curl -s http://127.0.0.1:8877/api/sessions/{session_id}/summary
```
Returns previously generated summary.

## Workflow

### During a Meeting
1. User joins a meeting (Google Meet, Zoom, Teams) in Chrome
2. Clicks ghostmeet extension icon → side panel opens
3. Clicks "Start" → real-time transcription begins
4. Transcripts appear live in the side panel

### After a Meeting
User asks: "Summarize my last meeting"

1. List sessions → find the latest session ID
2. Get transcript → review what was discussed
3. Generate summary → extract key points
4. Deliver summary to user

### Example Interaction

User: "What was discussed in my last meeting?"
→ `curl http://127.0.0.1:8877/api/sessions` → get latest session
→ `curl http://127.0.0.1:8877/api/sessions/{id}/transcript` → get transcript
→ Summarize key points for the user

User: "Generate a summary with action items"
→ `curl -X POST http://127.0.0.1:8877/api/sessions/{id}/summarize`
→ `curl http://127.0.0.1:8877/api/sessions/{id}/summary`
→ Deliver formatted summary

User: "How many meetings did I have today?"
→ `curl http://127.0.0.1:8877/api/sessions` → count today's sessions

## Configuration

| Variable | Default | Description |
|----------|---------|-------------|
| `GHOSTMEET_MODEL` | `base` | Whisper model (tiny/base/small/medium/large-v3) |
| `GHOSTMEET_LANGUAGE` | auto | Force language (en/ko/ja etc.) or auto-detect |
| `GHOSTMEET_CHUNK_INTERVAL` | `10` | Transcription interval in seconds |
| `GHOSTMEET_ANTHROPIC_KEY` | — | Claude API key for summaries |
| `GHOSTMEET_HOST` | `127.0.0.1` | Backend bind address (loopback — the API has no auth) |
| `GHOSTMEET_PORT` | `8877` | Backend port |

### Model Size Guide
- **tiny** (75MB): Fast, lower accuracy. Good for quick notes
- **base** (145MB): Balanced. Recommended for most users
- **small** (488MB): Better accuracy, slower
- **medium** (1.5GB): High accuracy, needs good CPU/GPU
- **large-v3** (3GB): Best accuracy, requires GPU

## Usage Guidelines

1. **Always check health first** — verify backend is running before other commands
2. **List sessions to find IDs** — session IDs are date-based (e.g., `20260308-065021`)
3. **Summarize only when asked** — summary generation costs API tokens
4. **Format transcripts nicely** — don't dump raw JSON, present as readable conversation
5. **Respect privacy** — meeting transcripts are sensitive. Never share outside the current chat
6. **If backend is down** — suggest `docker compose up -d` in the ghostmeet directory

## Privacy

- **Transcription is 100% local** — Whisper runs on your machine, audio never leaves your device
- **Summaries use Anthropic API** — when you click Summarize, transcript text is sent to Claude API. If you don't want this, skip the summarize feature; transcription works without it
- **Chrome Extension captures tab audio only** — uses `chrome.tabCapture` API, limited to the active tab. It cannot access other tabs, microphone, or system audio. Audit the extension source in `extension/` before installing
- **No telemetry** — ghostmeet sends zero analytics or tracking data

## Troubleshooting

- **Connection refused** → Backend not running. `docker compose up -d`
- **No sessions** → No meetings recorded yet. Chrome Extension must be active during meeting
- **Summary fails** → `GHOSTMEET_ANTHROPIC_KEY` not set in `.env`
- **Poor transcription** → Try larger Whisper model or set explicit language
