"""FastAPI application — web UI + API endpoints."""

import asyncio
import json
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect, Request
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from .config import config
from .meeting import MeetingBot
from .agent import MeetingAgent
from .tts import TTS

BASE_DIR = Path(__file__).parent

app = FastAPI(title="attend", version="0.1.0")
app.mount("/static", StaticFiles(directory=BASE_DIR / "static"), name="static")
templates = Jinja2Templates(directory=BASE_DIR / "templates")

# Active sessions: meeting_id -> session state
sessions: dict[str, dict] = {}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """Main page — paste a meeting link."""
    missing = config.validate()
    return templates.TemplateResponse("index.html", {
        "request": request,
        "missing_keys": missing,
        "config": config,
    })


@app.post("/api/join")
async def join_meeting(request: Request):
    """Join a meeting. Body: { "meeting_url": "https://zoom.us/j/..." }"""
    body = await request.json()
    meeting_url = body.get("meeting_url", "").strip()

    if not meeting_url:
        return {"error": "meeting_url is required"}

    missing = config.validate()
    if missing:
        return {"error": f"Missing API keys: {', '.join(missing)}"}

    bot = MeetingBot()
    agent = MeetingAgent()
    tts = TTS()

    try:
        bot_info = await bot.join(meeting_url)
    except Exception as e:
        return {"error": f"Failed to join meeting: {str(e)}"}

    bot_id = bot_info["id"]
    sessions[bot_id] = {
        "bot": bot,
        "agent": agent,
        "tts": tts,
        "meeting_url": meeting_url,
        "status": "joining",
        "transcript": [],
    }

    return {"bot_id": bot_id, "status": "joining"}


@app.get("/api/meeting/{bot_id}")
async def meeting_status(bot_id: str):
    """Get meeting status and transcript."""
    session = sessions.get(bot_id)
    if not session:
        return {"error": "Session not found"}

    bot: MeetingBot = session["bot"]
    status = await bot.status()

    return {
        "bot_id": bot_id,
        "status": status,
        "transcript": session["transcript"],
    }


@app.post("/api/meeting/{bot_id}/leave")
async def leave_meeting(bot_id: str):
    """Leave the meeting and generate summary."""
    session = sessions.get(bot_id)
    if not session:
        return {"error": "Session not found"}

    bot: MeetingBot = session["bot"]
    agent: MeetingAgent = session["agent"]

    # Generate summary before leaving
    summary = await agent.generate_summary()

    await bot.leave()
    session["status"] = "ended"

    return {"summary": summary}


@app.post("/api/webhook/transcript")
async def transcript_webhook(request: Request):
    """Webhook endpoint for Recall.ai real-time transcription."""
    body = await request.json()

    bot_id = body.get("bot_id")
    transcript_data = body.get("data", {})

    session = sessions.get(bot_id)
    if not session:
        return {"ok": False}

    agent: MeetingAgent = session["agent"]
    tts: TTS = session["tts"]
    bot: MeetingBot = session["bot"]

    speaker = transcript_data.get("speaker", "Unknown")
    text = transcript_data.get("transcript", "")

    if not text:
        return {"ok": True}

    # Add to transcript
    agent.add_utterance(speaker, text)
    session["transcript"].append({"speaker": speaker, "text": text})

    # Check if AI should respond
    if await agent.should_respond(text):
        response_text = await agent.generate_response()

        # Add AI response to transcript
        agent.add_utterance(config.bot_name, response_text)
        session["transcript"].append({
            "speaker": config.bot_name,
            "text": response_text,
            "is_ai": True,
        })

        # Convert to speech and send to meeting
        audio = await tts.synthesize(response_text)
        if audio:
            await bot.send_audio(audio)

    return {"ok": True}


@app.get("/meeting/{bot_id}", response_class=HTMLResponse)
async def meeting_page(request: Request, bot_id: str):
    """Live meeting dashboard."""
    session = sessions.get(bot_id)
    if not session:
        return HTMLResponse("<h1>Session not found</h1>", status_code=404)

    return templates.TemplateResponse("meeting.html", {
        "request": request,
        "bot_id": bot_id,
        "config": config,
    })


@app.websocket("/ws/meeting/{bot_id}")
async def meeting_ws(websocket: WebSocket, bot_id: str):
    """WebSocket for live transcript updates."""
    await websocket.accept()

    session = sessions.get(bot_id)
    if not session:
        await websocket.close()
        return

    last_idx = 0
    try:
        while True:
            transcript = session["transcript"]
            if len(transcript) > last_idx:
                new_entries = transcript[last_idx:]
                for entry in new_entries:
                    await websocket.send_json(entry)
                last_idx = len(transcript)

            # Also check bot status
            bot: MeetingBot = session["bot"]
            status = await bot.status()
            if status and status.get("status_changes"):
                latest_status = status["status_changes"][-1]["code"]
                await websocket.send_json({"type": "status", "status": latest_status})
                if latest_status == "done":
                    break

            await asyncio.sleep(1)
    except WebSocketDisconnect:
        pass
