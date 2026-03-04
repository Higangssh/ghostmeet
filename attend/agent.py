"""AI agent for meeting context understanding and response generation."""

import anthropic
from .config import config


class MeetingAgent:
    """Manages meeting context and generates AI responses."""

    def __init__(self):
        self.client = anthropic.AsyncAnthropic(api_key=config.anthropic_api_key)
        self.transcript: list[dict] = []  # [{"speaker": "...", "text": "..."}]
        self.summary: str = ""

    def add_utterance(self, speaker: str, text: str) -> None:
        """Add a new utterance to the transcript."""
        self.transcript.append({"speaker": speaker, "text": text})

    def _build_system_prompt(self) -> str:
        """Build system prompt with user context."""
        parts = [
            "You are an AI assistant attending a meeting on behalf of someone.",
            "You listen to the conversation and respond when appropriate.",
            "",
            "IMPORTANT RULES:",
            "- Be concise. Meeting responses should be 1-3 sentences max.",
            "- Only speak when directly asked a question or when your input is needed.",
            "- If unsure, say 'Let me check with [user] and get back to you.'",
            "- Never make commitments or decisions without noting them for the user.",
        ]

        if config.user_name:
            parts.append(f"\nYou are representing: {config.user_name}")
        if config.user_role:
            parts.append(f"Their role: {config.user_role}")
        if config.user_context:
            parts.append(f"Context they provided: {config.user_context}")

        return "\n".join(parts)

    def _format_transcript(self) -> str:
        """Format recent transcript for LLM context."""
        # Keep last 50 utterances to stay within context window
        recent = self.transcript[-50:]
        lines = []
        for u in recent:
            lines.append(f"{u['speaker']}: {u['text']}")
        return "\n".join(lines)

    async def should_respond(self, latest_text: str) -> bool:
        """Determine if the AI should respond to the latest utterance."""
        # Simple heuristics first
        bot_name = config.bot_name.lower()
        user_name = config.user_name.lower() if config.user_name else ""
        text_lower = latest_text.lower()

        # Direct mentions
        if bot_name in text_lower or (user_name and user_name in text_lower):
            return True

        # Question patterns
        question_words = ["?", "what do you think", "any thoughts", "can you", "could you"]
        if any(q in text_lower for q in question_words):
            return True

        return False

    async def generate_response(self) -> str:
        """Generate a response based on the current transcript."""
        transcript_text = self._format_transcript()

        response = await self.client.messages.create(
            model=config.anthropic_model,
            max_tokens=200,
            system=self._build_system_prompt(),
            messages=[
                {
                    "role": "user",
                    "content": f"Here is the meeting transcript so far:\n\n{transcript_text}\n\nGenerate a brief, appropriate response to the latest question or topic directed at you.",
                }
            ],
        )

        return response.content[0].text

    async def generate_summary(self) -> dict:
        """Generate a meeting summary with action items."""
        transcript_text = self._format_transcript()

        response = await self.client.messages.create(
            model=config.anthropic_model,
            max_tokens=1000,
            system="You are a meeting summarizer. Be concise and actionable.",
            messages=[
                {
                    "role": "user",
                    "content": f"Summarize this meeting transcript. Include:\n1. Key Discussion Points\n2. Decisions Made\n3. Action Items (with owners if mentioned)\n4. Follow-ups Needed\n\nTranscript:\n{transcript_text}",
                }
            ],
        )

        return {
            "summary": response.content[0].text,
            "utterance_count": len(self.transcript),
            "speakers": list(set(u["speaker"] for u in self.transcript)),
        }
