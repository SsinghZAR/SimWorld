"""Structured action schema for experimental multi-agent SimWorld turns."""

from __future__ import annotations

import json
from typing import Any

from pydantic import BaseModel

from simworld.local_planner.action_space import LowLevelAction, LowLevelActionSpace


DEFAULT_STEP_DURATION = 0.2
MAX_STEP_DURATION = 0.6
DEFAULT_TURN_ANGLE = 45.0
MAX_TURN_ANGLE = 180.0


class MultiAgentTurn(BaseModel):
    """One simultaneous movement and optional communication decision."""

    choice: int = LowLevelAction.DO_NOTHING.value
    duration: float | None = None
    direction: int | None = None
    angle: float | None = None
    clockwise: bool | None = None
    message: str | None = None
    reasoning: str | None = None

    @classmethod
    def from_json(cls, payload: Any) -> "MultiAgentTurn":
        """Parse a turn from a JSON string or dictionary."""

        if payload is None:
            return cls()
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
            # Future directed-message schema can fit here; for now use content if present.
            message = message.get("content")

        return cls(
            choice=int(payload.get("choice", LowLevelAction.DO_NOTHING.value) or LowLevelAction.DO_NOTHING.value),
            duration=payload.get("duration"),
            direction=payload.get("direction"),
            angle=payload.get("angle"),
            clockwise=payload.get("clockwise"),
            message=message.strip() if isinstance(message, str) and message.strip() else None,
            reasoning=payload.get("reasoning"),
        )

    @classmethod
    def to_json_schema(cls) -> dict[str, Any]:
        """Return the JSON schema used in MiniMax prompts."""

        return {
            "name": "MultiAgentTurn",
            "strict": True,
            "schema": {
                "type": "object",
                "properties": {
                    "choice": {
                        "type": "integer",
                        "description": "Movement action. 0=DO_NOTHING, 1=STEP_FORWARD, 2=TURN_AROUND.",
                    },
                    "duration": {
                        "type": ["number", "null"],
                        "description": "Seconds for STEP_FORWARD; use a small value like 0.2.",
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
                    "message": {
                        "type": ["string", "null"],
                        "description": "Optional short broadcast message to teammates. Use null if no useful message.",
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
        """Return a JSON-serializable dict compatible with pydantic v1/v2."""

        if hasattr(self, "model_dump"):
            return self.model_dump(mode="json")
        return self.dict()


def action_to_dict(action: LowLevelActionSpace) -> dict[str, Any]:
    """Return a pydantic action as a JSON-serializable dict."""

    if hasattr(action, "model_dump"):
        return action.model_dump(mode="json")
    return action.dict()


def sanitize_turn(
    turn: MultiAgentTurn,
    *,
    default_step_duration: float = DEFAULT_STEP_DURATION,
    max_step_duration: float = MAX_STEP_DURATION,
    default_turn_angle: float = DEFAULT_TURN_ANGLE,
    max_turn_angle: float = MAX_TURN_ANGLE,
    relative_angle: float = 0.0,
) -> LowLevelActionSpace:
    """Clamp a parsed turn into SimWorld's low-level movement envelope."""

    if turn.choice == LowLevelAction.STEP_FORWARD.value:
        duration = turn.duration if turn.duration and turn.duration > 0 else default_step_duration
        duration = max(0.05, min(float(duration), max_step_duration))
        direction = int(turn.direction) if turn.direction in (0, 1) else 0
        return LowLevelActionSpace(
            choice=LowLevelAction.STEP_FORWARD,
            duration=duration,
            direction=direction,
            angle=None,
            clockwise=None,
            reasoning=turn.reasoning,
        )

    if turn.choice == LowLevelAction.TURN_AROUND.value:
        angle = turn.angle if turn.angle and turn.angle > 0 else min(abs(relative_angle), default_turn_angle)
        angle = max(1.0, min(float(angle), max_turn_angle))
        clockwise = bool(turn.clockwise) if turn.clockwise is not None else relative_angle < 0
        return LowLevelActionSpace(
            choice=LowLevelAction.TURN_AROUND,
            duration=None,
            direction=None,
            angle=angle,
            clockwise=clockwise,
            reasoning=turn.reasoning,
        )

    return LowLevelActionSpace(choice=LowLevelAction.DO_NOTHING, reasoning=turn.reasoning)
