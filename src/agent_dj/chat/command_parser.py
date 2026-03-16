"""Command Parser — Parses mid-session chat messages into structured DJ commands."""

import json
import re
from dataclasses import dataclass

from anthropic import Anthropic

from .prompts import COMMAND_SYSTEM_PROMPT


@dataclass
class DJCommand:
    """A parsed command from a chat message."""
    type: str               # adjust_energy, adjust_genre, skip, request, wind_down, adjust_tempo, chat
    parameters: dict        # type-specific parameters
    raw_message: str        # original user message
    dj_response: str = ""   # DJ's response (for chat type)


class CommandParser:
    """Parse mid-session messages into structured commands via LLM."""

    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def parse(self, message: str) -> DJCommand:
        """Parse a user message into a DJCommand."""
        response = self.client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=200,
            system=COMMAND_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": message}],
        )

        response_text = response.content[0].text
        command_data = _extract_command_json(response_text)

        if not command_data:
            return DJCommand(
                type="chat",
                parameters={"response": "I didn't quite catch that — what would you like me to change?"},
                raw_message=message,
            )

        cmd_type = command_data.pop("type", "chat")
        dj_response = command_data.pop("response", "")

        return DJCommand(
            type=cmd_type,
            parameters=command_data,
            raw_message=message,
            dj_response=dj_response,
        )


def _extract_command_json(text: str) -> dict | None:
    """Extract JSON from the LLM response."""
    # Try code block first
    match = re.search(r"```json?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(1))
        except json.JSONDecodeError:
            pass

    # Try raw JSON
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            pass

    return None
