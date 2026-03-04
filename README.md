# ghostmeet

Self-hosted AI meeting delegate.

## Current status
- ✅ Step 1 complete: architecture/scope locked
- ✅ Step 2 complete: Chrome tab audio capture → local backend WebSocket ingest PoC
- ⏳ Step 3+: STT/summary/agent mode pending

See `IMPLEMENTATION_PLAN.md` and `requirements.md`.

---

## PoC (Step 2) run guide

### 1) Start backend
```bash
cd /Users/sanghee/.openclaw/workspace/projects/ghostmeet
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
python -m backend
```

Backend runs on `http://127.0.0.1:8877`.

- Health: `GET /api/health`
- Session stats: `GET /api/sessions`
- WebSocket ingest: `ws://127.0.0.1:8877/ws/audio?session=<id>`

### 2) Load Chrome extension
1. Open `chrome://extensions`
2. Enable **Developer mode**
3. Click **Load unpacked**
4. Select folder: `.../projects/ghostmeet/extension`

### 3) Capture test
1. Open Zoom/Meet tab in Chrome
2. Click extension icon
3. Press **Start capture**
4. Talk in meeting tab for ~10s
5. Press **Stop capture**
6. Check backend stats at `/api/sessions`
7. Verify file in `recordings/<session>.webm`

---

## Repo layout (current)

```text
ghostmeet/
├── extension/                 # Chrome MV3 extension (PoC)
│   ├── manifest.json
│   ├── background.js
│   ├── popup.html
│   └── popup.js
├── backend/                   # local FastAPI backend (PoC)
│   ├── __main__.py
│   └── app.py
├── recordings/                # captured webm chunks (runtime output)
├── IMPLEMENTATION_PLAN.md
├── requirements.md
└── requirements.txt
```
