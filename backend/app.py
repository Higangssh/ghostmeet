"""ghostmeet backend — audio capture + chunked STT."""
from __future__ import annotations

import asyncio
import datetime as dt
import logging
import os
import tempfile
from pathlib import Path
from typing import Dict, List

import uvicorn
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from .audio_processor import transcribe_webm_file
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
# chunk interval in seconds (default 5 minutes)
CHUNK_INTERVAL = int(os.environ.get("GHOSTMEET_CHUNK_INTERVAL", "300"))

app = FastAPI(title="ghostmeet-backend", version="0.3.0")

# CORS for local development (extension + demo pages)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# state
sessions: Dict[str, Session] = {}
transcribers: Dict[str, Transcriber] = {}
summaries: Dict[str, Summary] = {}
# websocket subscribers for live transcript
transcript_subscribers: Dict[str, List[WebSocket]] = {}


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "service": "ghostmeet-backend",
        "model": WHISPER_MODEL,
        "chunk_interval_sec": CHUNK_INTERVAL,
    }


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


async def _transcribe_chunk_file(
    chunk_path: Path,
    transcriber: Transcriber,
    session: Session,
    session_id: str,
):
    """Transcribe a chunk file and broadcast results."""
    logger.info("Transcribing chunk: %s", chunk_path.name)
    loop = asyncio.get_event_loop()
    new_segments = await loop.run_in_executor(
        None, transcribe_webm_file, chunk_path, transcriber
    )
    if new_segments:
        session.transcript_segments = len(transcriber.transcript)
        await _broadcast_segments(session_id, new_segments)
        logger.info("Chunk %s: %d new segments", chunk_path.name, len(new_segments))


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

    # chunk management
    chunk_dir = RECORDINGS_DIR / f"{session_id}_chunks"
    chunk_dir.mkdir(exist_ok=True)
    chunk_num = 0
    chunk_start_time = asyncio.get_event_loop().time()
    current_chunk_path = chunk_dir / f"chunk_{chunk_num:04d}.webm"
    current_chunk_file = current_chunk_path.open("wb")

    # background transcription tasks
    transcription_tasks: List[asyncio.Task] = []

    try:
        with out_path.open("ab") as full_file:
            while True:
                message = await websocket.receive()
                if "bytes" in message and message["bytes"]:
                    chunk = message["bytes"]

                    # write to full recording
                    full_file.write(chunk)
                    full_file.flush()

                    # write to current chunk
                    current_chunk_file.write(chunk)
                    current_chunk_file.flush()

                    session.chunks += 1
                    session.audio_bytes += len(chunk)

                    # check if chunk interval elapsed
                    elapsed = asyncio.get_event_loop().time() - chunk_start_time
                    if elapsed >= CHUNK_INTERVAL:
                        # close current chunk and start transcription
                        current_chunk_file.close()
                        task = asyncio.create_task(
                            _transcribe_chunk_file(
                                current_chunk_path, transcriber, session, session_id
                            )
                        )
                        transcription_tasks.append(task)

                        # start new chunk
                        chunk_num += 1
                        chunk_start_time = asyncio.get_event_loop().time()
                        current_chunk_path = chunk_dir / f"chunk_{chunk_num:04d}.webm"
                        current_chunk_file = current_chunk_path.open("wb")

                        session.status = "streaming"
                        logger.info(
                            "Session %s: chunk %d started (after %.0fs)",
                            session_id, chunk_num, elapsed,
                        )

                elif "text" in message and message["text"] == "stop":
                    break
    except WebSocketDisconnect:
        pass
    finally:
        # close and transcribe the last chunk
        current_chunk_file.close()

        if current_chunk_path.stat().st_size > 0:
            session.status = "transcribing"
            task = asyncio.create_task(
                _transcribe_chunk_file(
                    current_chunk_path, transcriber, session, session_id
                )
            )
            transcription_tasks.append(task)

        # wait for all transcription tasks to finish
        if transcription_tasks:
            logger.info("Waiting for %d transcription tasks...", len(transcription_tasks))
            try:
                await asyncio.wait_for(
                    asyncio.gather(*transcription_tasks, return_exceptions=True),
                    timeout=300,  # max 5 min wait
                )
            except asyncio.TimeoutError:
                logger.error("Transcription tasks timed out")

        session.status = "stopped"
        session.stopped_at = dt.datetime.now().isoformat(timespec="seconds")
        logger.info(
            "Session %s complete: %d chunks, %d bytes, %d segments, %d audio chunks",
            session_id, session.chunks, session.audio_bytes,
            session.transcript_segments, chunk_num + 1,
        )


def run() -> None:
    uvicorn.run(app, host="127.0.0.1", port=8877)
