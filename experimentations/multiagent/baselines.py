"""Scripted baselines for the multi-agent rendezvous environment."""

from __future__ import annotations

import math
import re
from typing import Any

from .action_space import MultiAgentTurn


TARGET_PATTERN = re.compile(r"target\s*=\s*\(?\s*(-?\d+(?:\.\d+)?)\s*,\s*(-?\d+(?:\.\d+)?)", re.IGNORECASE)


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-180, 180]."""

    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


def turn_toward(relative_angle: float, *, message: str | None = None, reasoning: str | None = None) -> MultiAgentTurn:
    """Return a small deterministic movement toward a relative target angle."""

    if abs(relative_angle) > 12:
        return MultiAgentTurn(
            choice=2,
            angle=min(abs(relative_angle), 45),
            clockwise=relative_angle < 0,
            message=message,
            reasoning=reasoning or "turning toward the rendezvous point",
        )
    return MultiAgentTurn(
        choice=1,
        duration=0.2,
        direction=0,
        message=message,
        reasoning=reasoning or "walking toward the rendezvous point",
    )


def parse_target_from_inbox(observation: dict[str, Any]) -> tuple[float, float] | None:
    """Extract a target coordinate broadcast from the agent inbox."""

    for message in reversed(observation.get("inbox", [])):
        content = str(message.get("content", ""))
        match = TARGET_PATTERN.search(content)
        if match:
            return float(match.group(1)), float(match.group(2))
    return None


def relative_angle_to(point: tuple[float, float], observation: dict[str, Any]) -> float:
    """Compute relative angle from observation pose to a target point."""

    position = observation["position"]
    dx = point[0] - float(position["x"])
    dy = point[1] - float(position["y"])
    target_yaw = math.degrees(math.atan2(dy, dx))
    return normalize_angle(target_yaw - float(observation["yaw_deg"]))


class SilentBaselinePolicy:
    """Move when the target is directly known; never send messages."""

    def act_all(self, observations: dict[str, dict[str, Any]], **_: Any) -> tuple[dict[str, MultiAgentTurn], list[dict[str, Any]]]:
        turns: dict[str, MultiAgentTurn] = {}
        records: list[dict[str, Any]] = []
        for agent_id, observation in observations.items():
            if observation.get("target_known"):
                turn = turn_toward(float(observation["relative_angle_deg"]), reasoning="silent baseline follows known target")
            else:
                turn = MultiAgentTurn(reasoning="silent baseline has no target information")
            turns[agent_id] = turn
            records.append({"agent_id": agent_id, "baseline": "silent", "parsed_turn": turn.compact()})
        return turns, records


class CommunicatingBaselinePolicy:
    """Agent 0 broadcasts the target; others follow broadcasts when received."""

    def act_all(self, observations: dict[str, dict[str, Any]], **_: Any) -> tuple[dict[str, MultiAgentTurn], list[dict[str, Any]]]:
        turns: dict[str, MultiAgentTurn] = {}
        records: list[dict[str, Any]] = []
        for agent_id, observation in observations.items():
            message = None
            if observation.get("target_known"):
                target = observation["target"]
                if agent_id == "agent_0":
                    message = f"target=({target['x']:.1f},{target['y']:.1f}); rendezvous there"
                turn = turn_toward(
                    float(observation["relative_angle_deg"]),
                    message=message,
                    reasoning="communicating baseline follows known target",
                )
            else:
                target_from_inbox = parse_target_from_inbox(observation)
                if target_from_inbox is None:
                    turn = MultiAgentTurn(reasoning="waiting for target broadcast")
                else:
                    turn = turn_toward(
                        relative_angle_to(target_from_inbox, observation),
                        reasoning="following target broadcast",
                    )
            turns[agent_id] = turn
            records.append({"agent_id": agent_id, "baseline": "communicating", "parsed_turn": turn.compact()})
        return turns, records
