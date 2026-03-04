"""Recall.ai meeting bot management."""

import httpx
from .config import config


class MeetingBot:
    """Manages a Recall.ai bot for a single meeting."""

    def __init__(self):
        self.api_url = config.recall_api_url
        self.headers = {
            "Authorization": f"Token {config.recall_api_key}",
            "Content-Type": "application/json",
        }
        self.bot_id: str | None = None

    async def join(self, meeting_url: str) -> dict:
        """Send bot to join a meeting. Returns bot info."""
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.api_url}/bot",
                headers=self.headers,
                json={
                    "meeting_url": meeting_url,
                    "bot_name": config.bot_name,
                    "transcription_options": {
                        "provider": "deepgram",
                    },
                    "real_time_transcription": {
                        "destination_url": "",  # Will be set via webhook
                        "partial_results": False,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            data = resp.json()
            self.bot_id = data["id"]
            return data

    async def leave(self) -> None:
        """Remove bot from the meeting."""
        if not self.bot_id:
            return
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.api_url}/bot/{self.bot_id}/leave_call",
                headers=self.headers,
                timeout=10,
            )

    async def status(self) -> dict | None:
        """Get bot status."""
        if not self.bot_id:
            return None
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.api_url}/bot/{self.bot_id}",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()

    async def send_audio(self, audio_data: bytes) -> None:
        """Stream audio to the meeting (bot speaks)."""
        if not self.bot_id:
            return
        async with httpx.AsyncClient() as client:
            await client.post(
                f"{self.api_url}/bot/{self.bot_id}/output_audio",
                headers={
                    "Authorization": f"Token {config.recall_api_key}",
                    "Content-Type": "audio/raw",
                },
                content=audio_data,
                timeout=30,
            )

    async def get_transcript(self) -> list[dict]:
        """Get current transcript."""
        if not self.bot_id:
            return []
        async with httpx.AsyncClient() as client:
            resp = await client.get(
                f"{self.api_url}/bot/{self.bot_id}/transcript",
                headers=self.headers,
                timeout=10,
            )
            resp.raise_for_status()
            return resp.json()
