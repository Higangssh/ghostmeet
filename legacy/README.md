# legacy — ghostmeet v0.1

The v0.1 line, kept as it was tagged (`v0.1.0`): a Chrome MV3 extension that streams tab
audio to a local FastAPI backend (faster-whisper), with an OpenClaw skill on top. The
desktop app (`app/`, `crates/`) replaces it from 0.2; nothing new is built here. The
benchmark (`bench/`) still imports `backend/` to measure the v0.1 tail window.

```bash
cd legacy
python -m venv ../.venv && ../.venv/bin/python -m pip install -r requirements-dev.txt
../.venv/bin/python -m pytest tests/ -q          # 112 tests, no model, no network
node --test tests/extension/shared.test.mjs     # extension helpers
../.venv/bin/python -m backend                   # backend on 127.0.0.1:8877
docker compose up -d                             # or in Docker, published on loopback only
```

Run from `legacy/`: the backend now keeps its recordings in `legacy/recordings/`
(gitignored). Recordings an older checkout made in the repository root are not moved.
