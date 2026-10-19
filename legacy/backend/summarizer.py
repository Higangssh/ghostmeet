"""Meeting summary generation using Claude API."""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Optional

import httpx

logger = logging.getLogger(__name__)

ANTHROPIC_API_URL = "https://api.anthropic.com/v1/messages"

SUMMARY_PROMPT = """You are a meeting summarizer. Analyze the following meeting transcript and produce a structured summary.

Output format (use exactly this structure):

## 📋 Meeting Summary

### 📌 Key Decisions
- [List each decision made during the meeting]

### ✅ Action Items
- [Person/Role]: [Task] ([Deadline if mentioned])

### 🔑 Key Discussion Points
- [Main topics discussed with brief context]

### ❓ Open Questions
- [Unresolved items or questions raised but not answered]

### 📝 Brief Summary
[2-3 sentence overview of the meeting]

Rules:
- Be concise and actionable
- If no decisions/actions/questions exist, write "None identified"
- Use the original language of the transcript (don't translate)
- Focus on what matters, skip small talk

Transcript:
---
{transcript}
---"""


@dataclass
class Summary:
    """Generated meeting summary."""
    session_id: str
    content: str = ""
    model: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    status: str = "pending"  # pending | generating | done | error
    error: Optional[str] = None

    def to_dict(self):
        return {
            "session_id": self.session_id,
            "content": self.content,
            "model": self.model,
            "input_tokens": self.input_tokens,
            "output_tokens": self.output_tokens,
            "status": self.status,
            "error": self.error,
        }


async def generate_summary(
    transcript_text: str,
    session_id: str,
    api_key: Optional[str] = None,
    model: str = "claude-sonnet-4-20250514",
) -> Summary:
    """Generate a meeting summary from transcript text."""
    summary = Summary(session_id=session_id)

    key = api_key or os.environ.get("GHOSTMEET_ANTHROPIC_KEY")
    if not key:
        summary.status = "error"
        summary.error = "GHOSTMEET_ANTHROPIC_KEY not set"
        return summary

    if not transcript_text.strip():
        summary.status = "error"
        summary.error = "Empty transcript"
        return summary

    summary.status = "generating"
    prompt = SUMMARY_PROMPT.format(transcript=transcript_text)

    try:
        async with httpx.AsyncClient(timeout=60.0) as client:
            resp = await client.post(
                ANTHROPIC_API_URL,
                headers={
                    "x-api-key": key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json",
                },
                json={
                    "model": model,
                    "max_tokens": 2048,
                    "messages": [
                        {"role": "user", "content": prompt},
                    ],
                },
            )
            resp.raise_for_status()
            data = resp.json()

            summary.content = data["content"][0]["text"]
            summary.model = data.get("model", model)
            summary.input_tokens = data.get("usage", {}).get("input_tokens", 0)
            summary.output_tokens = data.get("usage", {}).get("output_tokens", 0)
            summary.status = "done"

            logger.info(
                "Summary generated for %s: %d input / %d output tokens",
                session_id, summary.input_tokens, summary.output_tokens,
            )

    except httpx.HTTPStatusError as e:
        summary.status = "error"
        summary.error = f"API error: {e.response.status_code} {e.response.text[:200]}"
        logger.error("Claude API error: %s", summary.error)
    except Exception as e:
        summary.status = "error"
        summary.error = f"Request failed: {str(e)}"
        logger.error("Summary generation failed: %s", e)

    return summary
