"""Configuration management for attend."""

import os
from dataclasses import dataclass, field
from pathlib import Path

from dotenv import load_dotenv

# Load .env from project root
_env_path = Path(__file__).parent.parent / ".env"
load_dotenv(_env_path)


@dataclass
class Config:
    # Recall.ai
    recall_api_key: str = ""
    recall_api_url: str = "https://us-west-2.recall.ai/api/v1"

    # Anthropic
    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    # ElevenLabs
    elevenlabs_api_key: str = ""
    elevenlabs_voice_id: str = "21m00Tcm4TlvDq8ikWAM"  # Rachel (default)

    # Bot settings
    bot_name: str = "AI Assistant"
    user_name: str = ""
    user_role: str = ""
    user_context: str = ""

    # Server
    host: str = "127.0.0.1"
    port: int = 8877

    @classmethod
    def from_env(cls) -> "Config":
        return cls(
            recall_api_key=os.getenv("RECALL_API_KEY", ""),
            recall_api_url=os.getenv("RECALL_API_URL", "https://us-west-2.recall.ai/api/v1"),
            anthropic_api_key=os.getenv("ANTHROPIC_API_KEY", ""),
            anthropic_model=os.getenv("ANTHROPIC_MODEL", "claude-sonnet-4-20250514"),
            elevenlabs_api_key=os.getenv("ELEVENLABS_API_KEY", ""),
            elevenlabs_voice_id=os.getenv("ELEVENLABS_VOICE_ID", "21m00Tcm4TlvDq8ikWAM"),
            bot_name=os.getenv("BOT_NAME", "AI Assistant"),
            user_name=os.getenv("USER_NAME", ""),
            user_role=os.getenv("USER_ROLE", ""),
            user_context=os.getenv("USER_CONTEXT", ""),
            host=os.getenv("HOST", "127.0.0.1"),
            port=int(os.getenv("PORT", "8877")),
        )

    def validate(self) -> list[str]:
        """Return list of missing required keys."""
        missing = []
        if not self.recall_api_key:
            missing.append("RECALL_API_KEY")
        if not self.anthropic_api_key:
            missing.append("ANTHROPIC_API_KEY")
        return missing


config = Config.from_env()
