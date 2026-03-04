from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Dict, Any

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

app = FastAPI(title="ghostmeet-backend", version="0.1.0")

# in-memory session stats for PoC
sessions: Dict[str, Dict[str, Any]] = {}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ghostmeet-backend"}


@app.get("/api/sessions")
def list_sessions():
    return {
        "count": len(sessions),
        "sessions": sessions,
    }


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    session_id = websocket.query_params.get("session")
    if not session_id:
        session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    out_path = RECORDINGS_DIR / f"{session_id}.webm"
    sessions.setdefault(
        session_id,
        {
            "chunks": 0,
            "bytes": 0,
            "started_at": dt.datetime.now().isoformat(timespec="seconds"),
            "file": str(out_path.relative_to(ROOT)),
            "status": "streaming",
        },
    )

    try:
        with out_path.open("ab") as f:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    chunk = message["bytes"]
                    f.write(chunk)
                    sessions[session_id]["chunks"] += 1
                    sessions[session_id]["bytes"] += len(chunk)
                elif "text" in message and message["text"] == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        sessions[session_id]["status"] = "stopped"
        sessions[session_id]["stopped_at"] = dt.datetime.now().isoformat(timespec="seconds")


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8877)
