# attend

AI meeting delegate (self-hosted, open source).

## Modes
- Ghost Mode: Join + transcribe + summarize
- Agent Mode: Join + transcribe + answer in-meeting (TTS)

## Quick start

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
python -m attend
```

Open: http://127.0.0.1:8877

## Required keys
- `RECALL_API_KEY` (required)
- `ANTHROPIC_API_KEY` (required)
- `ELEVENLABS_API_KEY` (optional, only for Agent Mode voice)

## Status
Prototype in progress (v0.1.0-dev)
