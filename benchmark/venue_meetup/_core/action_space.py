"""Structured turn schema for Venue Meetup agents."""

from __future__ import annotations

import json
from enum import Enum
from typing import Any

from pydantic import BaseModel, Field

DEFAULT_STEP_DURATION = 0.25
MAX_STEP_DURATION = 0.75
DEFAULT_TURN_ANGLE = 45.0
MAX_TURN_ANGLE = 180.0


class VenueAction(Enum):
    """Venue meetup action ids."""

    WAIT = 0
    STEP_FORWARD = 1
    TURN_AROUND = 2
    INSPECT = 3
    COMMUNICATE = 4
    NAVIGATE = 5


class SharedFactClaim(BaseModel):
    """One machine-readable fact claim attached to a turn or message."""

    venue_id: str
    trait: str
    value: Any

    def compact(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


def parse_shared_facts(raw: Any) -> list[SharedFactClaim]:
    """Parse shared_facts without raising on malformed or unsupported entries.

    Unknown traits/venues are retained for offline analysis rather than rejected.
    Completely malformed items are skipped.
    """

    if raw is None:
        return []
    if not isinstance(raw, list):
        return []

    claims: list[SharedFactClaim] = []
    for item in raw:
        if not isinstance(item, dict):
            continue
        venue_id = item.get("venue_id")
        trait = item.get("trait")
        if venue_id is None or trait is None or "value" not in item:
            continue
        venue_text = str(venue_id).strip()
        trait_text = str(trait).strip()
        if not venue_text or not trait_text:
            continue
        claims.append(SharedFactClaim(venue_id=venue_text, trait=trait_text, value=item["value"]))
    return claims


class VenueAgentTurn(BaseModel):
    """One venue-meetup turn emitted by a model or scripted policy."""

    choice: int = VenueAction.WAIT.value
    duration: float | None = None
    direction: int | None = None
    angle: float | None = None
    clockwise: bool | None = None
    target_venue_id: str | None = None
    target_description: str | None = None
    message: str | None = None
    shared_facts: list[SharedFactClaim] = Field(default_factory=list)
    reasoning: str | None = None

    @classmethod
    def from_json(cls, payload: Any) -> "VenueAgentTurn":
        """Parse a turn from a JSON string or dictionary."""

        if payload is None:
            return cls(reasoning="empty model response")
        if isinstance(payload, cls):
            return payload
        if isinstance(payload, str):
            try:
                payload = json.loads(payload)
            except json.JSONDecodeError:
                return cls(reasoning="failed to parse model JSON")
        if not isinstance(payload, dict):
            return cls(reasoning="model response was not a JSON object")

        message = payload.get("message")
        if isinstance(message, dict):
            message = message.get("content")

        target_description = payload.get("target_description")
        if target_description is not None:
            target_description = str(target_description).strip() or None

        target_venue_id = payload.get("target_venue_id")
        if target_venue_id is not None:
            target_venue_id = str(target_venue_id).strip() or None

        return cls(
            choice=int(payload.get("choice", VenueAction.WAIT.value) or VenueAction.WAIT.value),
            duration=payload.get("duration"),
            direction=payload.get("direction"),
            angle=payload.get("angle"),
            clockwise=payload.get("clockwise"),
            target_venue_id=target_venue_id,
            target_description=target_description,
            message=message.strip() if isinstance(message, str) and message.strip() else None,
            shared_facts=parse_shared_facts(payload.get("shared_facts")),
            reasoning=payload.get("reasoning"),
        )

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Return the JSON schema used in prompts."""

        return {
            "name": "VenueAgentTurn",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "integer",
                        "description": "0=WAIT, 1=STEP_FORWARD, 2=TURN_AROUND, 3=INSPECT visible/near venue, 4=COMMUNICATE only, 5=NAVIGATE to target_venue_id.",
                    },
                    "duration": {
                        "type": ["number", "null"],
                        "description": "Seconds for STEP_FORWARD; use a small value like 0.25.",
                    },
                    "direction": {
                        "type": ["integer", "null"],
                        "description": "STEP_FORWARD direction. 0=forward, 1=backward.",
                    },
                    "angle": {
                        "type": ["number", "null"],
                        "description": "TURN_AROUND angle in degrees, from 1 to 180.",
                    },
                    "clockwise": {
                        "type": ["boolean", "null"],
                        "description": "TURN_AROUND direction. true=right/clockwise, false=left/counterclockwise.",
                    },
                    "target_venue_id": {
                        "type": ["string", "null"],
                        "description": "Venue id for INSPECT or NAVIGATE when known from coarse map or previous observations.",
                    },
                    "target_description": {
                        "type": ["string", "null"],
                        "description": "Short visible target description for INSPECT when unsure of id.",
                    },
                    "message": {
                        "type": ["string", "null"],
                        "description": "Optional short broadcast message to teammates. For choice=4, provide message and/or shared_facts.",
                    },
                    "shared_facts": {
                        "type": ["array", "null"],
                        "description": (
                            "Optional structured fact claims for teammates. Include only traits you personally "
                            "INSPECTed. Each item is {venue_id, trait, value}. Unsupported/unknown traits are "
                            "logged for analysis; free-text message remains optional and separate."
                        ),
                        "items": {
                            "type": "object",
                            "properties": {
                                "venue_id": {"type": "string"},
                                "trait": {"type": "string"},
                                "value": {
                                    "description": "JSON value for the trait (boolean, number, or string).",
                                },
                            },
                            "required": ["venue_id", "trait", "value"],
                        },
                    },
                    "reasoning": {
                        "type": ["string", "null"],
                        "description": "One short sentence explaining the action.",
                    },
                },
                "required": ["choice"],
            },
        }

    def compact(self) -> dict[str, Any]:
        """Return a JSON-serializable representation."""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


def sanitize_turn(turn: VenueAgentTurn, *, relative_angle: float = 0.0) -> VenueAgentTurn:
    """Clamp model output to the benchmark action envelope."""

    shared_facts = list(turn.shared_facts or [])

    if turn.choice == VenueAction.STEP_FORWARD.value:
        duration = turn.duration if turn.duration and turn.duration > 0 else DEFAULT_STEP_DURATION
        duration = max(0.05, min(float(duration), MAX_STEP_DURATION))
        direction = int(turn.direction) if turn.direction in (0, 1) else 0
        return VenueAgentTurn(
            choice=VenueAction.STEP_FORWARD.value,
            duration=duration,
            direction=direction,
            message=turn.message,
            shared_facts=shared_facts,
            reasoning=turn.reasoning,
        )

    if turn.choice == VenueAction.TURN_AROUND.value:
        angle = turn.angle if turn.angle and turn.angle > 0 else min(abs(relative_angle), DEFAULT_TURN_ANGLE)
        angle = max(1.0, min(float(angle), MAX_TURN_ANGLE))
        clockwise = bool(turn.clockwise) if turn.clockwise is not None else relative_angle < 0
        return VenueAgentTurn(
            choice=VenueAction.TURN_AROUND.value,
            angle=angle,
            clockwise=clockwise,
            message=turn.message,
            shared_facts=shared_facts,
            reasoning=turn.reasoning,
        )

    if turn.choice == VenueAction.INSPECT.value:
        return VenueAgentTurn(
            choice=VenueAction.INSPECT.value,
            target_venue_id=turn.target_venue_id,
            target_description=turn.target_description,
            message=turn.message,
            shared_facts=shared_facts,
            reasoning=turn.reasoning,
        )

    if turn.choice == VenueAction.COMMUNICATE.value:
        return VenueAgentTurn(
            choice=VenueAction.COMMUNICATE.value,
            message=turn.message,
            shared_facts=shared_facts,
            reasoning=turn.reasoning,
        )

    if turn.choice == VenueAction.NAVIGATE.value:
        return VenueAgentTurn(
            choice=VenueAction.NAVIGATE.value,
            target_venue_id=turn.target_venue_id,
            target_description=turn.target_description,
            message=turn.message,
            shared_facts=shared_facts,
            reasoning=turn.reasoning,
        )

    return VenueAgentTurn(
        choice=VenueAction.WAIT.value,
        message=turn.message,
        shared_facts=shared_facts,
        reasoning=turn.reasoning,
    )
