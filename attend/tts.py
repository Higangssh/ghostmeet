"""Text-to-speech via ElevenLabs API."""

import httpx
from .config import config


class TTS:
    """Generate speech audio from text using ElevenLabs."""

    API_URL = "https://api.elevenlabs.io/v1"

    def __init__(self):
        self.api_key = config.elevenlabs_api_key
        self.voice_id = config.elevenlabs_voice_id

    async def synthesize(self, text: str) -> bytes | None:
        """Convert text to speech. Returns raw audio bytes (mp3)."""
        if not self.api_key:
            return None

        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self.API_URL}/text-to-speech/{self.voice_id}",
                headers={
                    "xi-api-key": self.api_key,
                    "Content-Type": "application/json",
                },
                json={
                    "text": text,
                    "model_id": "eleven_turbo_v2_5",
                    "voice_settings": {
                        "stability": 0.5,
                        "similarity_boost": 0.75,
                    },
                },
                timeout=30,
            )
            resp.raise_for_status()
            return resp.content
