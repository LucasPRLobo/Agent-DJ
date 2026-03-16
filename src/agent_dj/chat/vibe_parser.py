"""Vibe Parser — LLM-powered onboarding conversation to build a VibeProfile."""

import json
import re
from dataclasses import dataclass, field

from anthropic import Anthropic

from ..planner.vibe_profile import EnergyArcType, VibeProfile, generate_energy_arc
from .prompts import ONBOARDING_SYSTEM_PROMPT


@dataclass
class OnboardingSession:
    """Tracks the state of an onboarding conversation."""
    messages: list[dict] = field(default_factory=list)
    is_complete: bool = False
    vibe_profile: VibeProfile | None = None


class VibeParser:
    """Manages the onboarding chat and extracts a VibeProfile via LLM."""

    def __init__(self, api_key: str | None = None):
        self.client = Anthropic(api_key=api_key) if api_key else Anthropic()

    def start_session(self) -> tuple[OnboardingSession, str]:
        """Start a new onboarding session. Returns the DJ's opening message."""
        session = OnboardingSession()

        # Get the DJ's opening message
        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=300,
            system=ONBOARDING_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": "Hey, I need a DJ for tonight!"}],
        )

        dj_message = response.content[0].text
        session.messages = [
            {"role": "user", "content": "Hey, I need a DJ for tonight!"},
            {"role": "assistant", "content": dj_message},
        ]

        return session, dj_message

    def send_message(self, session: OnboardingSession, user_message: str) -> str:
        """Send a user message and get the DJ's response.

        If the DJ has enough info, it will output a JSON block and the
        session will be marked as complete with a VibeProfile.
        """
        session.messages.append({"role": "user", "content": user_message})

        response = self.client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=500,
            system=ONBOARDING_SYSTEM_PROMPT,
            messages=session.messages,
        )

        dj_response = response.content[0].text
        session.messages.append({"role": "assistant", "content": dj_response})

        # Check if the response contains a JSON vibe profile
        vibe_data = _extract_json(dj_response)
        if vibe_data and vibe_data.get("ready"):
            session.vibe_profile = _build_vibe_profile(vibe_data)
            session.is_complete = True

        return dj_response


def _extract_json(text: str) -> dict | None:
    """Extract a JSON block from the LLM response."""
    # Try to find JSON in code blocks
    json_match = re.search(r"```json?\s*\n?(.*?)\n?```", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            pass

    # Try to find raw JSON
    json_match = re.search(r"\{[^{}]*\"ready\"[^{}]*\}", text, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(0))
        except json.JSONDecodeError:
            pass

    return None


def _build_vibe_profile(data: dict) -> VibeProfile:
    """Convert LLM JSON output to a VibeProfile."""
    arc_type_str = data.get("energy_arc_type", "build_and_drop")
    try:
        arc_type = EnergyArcType(arc_type_str)
    except ValueError:
        arc_type = EnergyArcType.BUILD_AND_DROP

    bpm_range = tuple(data.get("bpm_range", [90, 130]))

    return VibeProfile(
        genres=data.get("genres", {}),
        bpm_range=bpm_range,
        energy_arc=generate_energy_arc(arc_type),
        energy_arc_type=arc_type,
        mood_happy=data.get("mood_happy", 0.5),
        mood_aggressive=data.get("mood_aggressive", 0.3),
        mood_relaxed=data.get("mood_relaxed", 0.5),
        mood_sad=data.get("mood_sad", 0.1),
        danceability_min=data.get("danceability_min", 0.3),
        mood_tags=data.get("mood_tags", []),
        duration_minutes=data.get("duration_minutes", 60),
        transition_style=data.get("transition_style", "smooth"),
        avoid_genres=data.get("avoid_genres", []),
        example_tracks=data.get("example_tracks", []),
    )
