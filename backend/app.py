"""ghostmeet backend — audio capture + real-time STT."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException

from .audio_processor import webm_chunks_to_pcm
from .models import Session
from .summarizer import Summary, generate_summary
from .transcriber import Transcriber, Segment

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(name)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
RECORDINGS_DIR = ROOT / "recordings"
RECORDINGS_DIR.mkdir(parents=True, exist_ok=True)

# config from env
WHISPER_MODEL = os.environ.get("GHOSTMEET_MODEL", "base")
WHISPER_DEVICE = os.environ.get("GHOSTMEET_DEVICE", "auto")
WHISPER_LANGUAGE = os.environ.get("GHOSTMEET_LANGUAGE", None)

app = FastAPI(title="ghostmeet-backend", version="0.2.0")

# state
sessions: Dict[str, Session] = {}
transcribers: Dict[str, Transcriber] = {}
summaries: Dict[str, Summary] = {}
# websocket subscribers for live transcript
transcript_subscribers: Dict[str, List[WebSocket]] = {}


@app.get("/api/health")
def health():
    return {"ok": True, "service": "ghostmeet-backend", "model": WHISPER_MODEL}


@app.get("/api/sessions")
def list_sessions():
    return {
        "count": len(sessions),
        "sessions": {k: v.to_dict() for k, v in sessions.items()},
    }


@app.get("/api/sessions/{session_id}")
def get_session(session_id: str):
    if session_id not in sessions:
        raise HTTPException(status_code=404, detail="session not found")
    return sessions[session_id].to_dict()


@app.get("/api/sessions/{session_id}/transcript")
def get_transcript(session_id: str):
    if session_id not in transcribers:
        raise HTTPException(status_code=404, detail="session not found")
    t = transcribers[session_id]
    return {
        "session_id": session_id,
        "segments": t.get_full_transcript(),
        "full_text": t.get_full_text(),
        "segment_count": len(t.transcript),
    }


@app.post("/api/sessions/{session_id}/summarize")
async def summarize_session(session_id: str):
    if session_id not in transcribers:
        raise HTTPException(status_code=404, detail="session not found")

    t = transcribers[session_id]
    text = t.get_full_text()
    if not text.strip():
        raise HTTPException(status_code=400, detail="transcript is empty")

    summary = await generate_summary(text, session_id)
    summaries[session_id] = summary
    return summary.to_dict()


@app.get("/api/sessions/{session_id}/summary")
def get_summary(session_id: str):
    if session_id not in summaries:
        raise HTTPException(status_code=404, detail="summary not found — call POST /summarize first")
    return summaries[session_id].to_dict()


@app.websocket("/ws/transcript/{session_id}")
async def ws_transcript(websocket: WebSocket, session_id: str):
    """Subscribe to live transcript updates for a session."""
    await websocket.accept()
    transcript_subscribers.setdefault(session_id, []).append(websocket)
    try:
        while True:
            await websocket.receive_text()  # keep alive
    except WebSocketDisconnect:
        pass
    finally:
        transcript_subscribers.get(session_id, []).remove(websocket)


async def _broadcast_segments(session_id: str, segments: List[Segment]):
    """Push new segments to all transcript subscribers."""
    subs = transcript_subscribers.get(session_id, [])
    data = [s.to_dict() for s in segments]
    dead = []
    for ws in subs:
        try:
            await ws.send_json({"type": "transcript", "segments": data})
        except Exception:
            dead.append(ws)
    for ws in dead:
        subs.remove(ws)


@app.websocket("/ws/audio")
async def ws_audio(websocket: WebSocket):
    await websocket.accept()

    session_id = websocket.query_params.get("session")
    if not session_id:
        session_id = dt.datetime.now().strftime("%Y%m%d-%H%M%S")

    out_path = RECORDINGS_DIR / f"{session_id}.webm"
    session = Session(
        session_id=session_id,
        file=str(out_path.relative_to(ROOT)),
    )
    sessions[session_id] = session

    # init transcriber for this session
    transcriber = Transcriber(
        model_size=WHISPER_MODEL,
        device=WHISPER_DEVICE,
        language=WHISPER_LANGUAGE,
    )
    transcribers[session_id] = transcriber

    # queue for feeding audio to ffmpeg
    audio_queue: asyncio.Queue[bytes | None] = asyncio.Queue()

    async def _process_audio():
        """Convert webm→PCM and transcribe in real-time."""
        try:
            async for pcm_chunk in webm_chunks_to_pcm(audio_queue):
                new_segments = transcriber.transcribe_chunk(pcm_chunk)
                if new_segments:
                    session.transcript_segments = len(transcriber.transcript)
                    await _broadcast_segments(session_id, new_segments)
        except Exception as e:
            logger.error("STT pipeline error: %s", e)

    processor_task = asyncio.create_task(_process_audio())

    try:
        with out_path.open("ab") as f:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    chunk = message["bytes"]
                    f.write(chunk)
                    session.chunks += 1
                    session.audio_bytes += len(chunk)
                    await audio_queue.put(chunk)
                elif "text" in message and message["text"] == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        # signal EOF to ffmpeg
        await audio_queue.put(None)
        # wait for processing to finish (max 10s)
        try:
            await asyncio.wait_for(processor_task, timeout=10.0)
        except asyncio.TimeoutError:
            processor_task.cancel()

        session.status = "stopped"
        session.stopped_at = dt.datetime.now().isoformat(timespec="seconds")
        logger.info(
            "Session %s stopped: %d chunks, %d bytes, %d transcript segments",
            session_id, session.chunks, session.audio_bytes, session.transcript_segments,
        )


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8877)
