"""Prompt construction for Venue Meetup agents."""

from __future__ import annotations

import json
from typing import Any

VENUE_MEETUP_SYSTEM_PROMPT = """You are one visitor agent in a shared embodied SimWorld "venue meetup" task.
Your team must agree on the single best venue that is feasible for EVERYONE, then physically meet there.

What you receive each turn:
- A camera image (THIRD-PERSON: you see your own back, so it is unreliable for left/right).
- A coarse schematic map and your compass heading ("self_pose").
- "candidate_venues": every venue on the map. Under spatial partition each shows "zone_id" and "can_inspect": you can only INSPECT venues in your OWN area; for venues with can_inspect=false you must rely on a teammate's report.
- "known_venue_facts": structured ground-truth traits you have personally learned by inspecting (open, accessible, food_drink, quiet, uncrowded, shelter, near_transit, capacity).
- "private_constraint": YOUR hard requirement. Teammates have their own (possibly different) hard requirements you will not know unless they tell you.
- "group_chat": messages from teammates.

Action choices (set "choice"):
- 5=NAVIGATE target_venue_id: travel to a venue's meeting point in one action. Use this to move and to converge; it walks you there, so you do NOT need to micro-steer.
- 3=INSPECT target_venue_id: learn a venue's structured facts. You must be in its area (can_inspect=true). Inspect before trusting any trait.
- 4=COMMUNICATE message: send concise, factual findings to teammates.
- 1=STEP_FORWARD / 2=TURN_AROUND: optional fine movement; rarely needed if you NAVIGATE. For TURN_AROUND set angle (deg) and clockwise.
- 0=WAIT.

Coordination (this is what is being measured):
- Relevant venues are split across areas, so no single agent can verify the best venue alone: pool inspections and rely on teammate reports.
- Share facts your TEAMMATES need, not only the ones you personally care about. A trait you do not require may be exactly what another agent's constraint needs.
- State your own hard requirement so teammates can rule venues in or out for you.
- Once the group identifies the venue feasible for everyone, all NAVIGATE there.

Return ONLY one valid JSON object, no markdown and no prose."""


def strip_frame(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an observation copy without image arrays for prompts/logs."""

    return {key: value for key, value in observation.items() if key != "ego_view"}


def build_agent_prompt(observation: dict[str, Any]) -> str:
    """Build one agent's structured user prompt."""

    prompt_payload = strip_frame(observation)
    return (
        "Task: pool inspections with your teammates and converge on the single venue that is feasible for EVERYONE.\n"
        "Use NAVIGATE (choice=5) with target_venue_id to travel to a venue; it walks you there. INSPECT (choice=3) a venue in your own area to learn its structured facts.\n"
        "Only inspect venues with can_inspect=true; for the rest, rely on teammate reports and share what your teammates need (not only what you need).\n"
        "Do not infer traits from metadata - INSPECT to learn them, then COMMUNICATE the facts that matter to others.\n"
        "Your observation JSON follows.\n"
        f"{json.dumps(prompt_payload, indent=2, default=str)}\n"
        "Return JSON keys: choice, duration, direction, angle, clockwise, target_venue_id, target_description, message, reasoning.\n"
        "Keep message short and factual. Include target_venue_id for NAVIGATE and INSPECT when known."
    )
