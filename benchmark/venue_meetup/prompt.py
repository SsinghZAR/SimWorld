"""Prompt construction for Venue Meetup agents.

The benchmark has two deliberately small prompt modes. Both modes use the
same task/action contract; ``cooperative`` only adds an explicit strategy
scaffold. Keeping the contract in one place makes it harder for the prompt
and action schema to drift apart while preserving a minimally instructed
baseline for social evaluation.
"""

from __future__ import annotations

import json
from typing import Any, Literal

PromptMode = Literal["minimal", "cooperative"]
PROMPT_MODES: tuple[str, str] = ("minimal", "cooperative")


def normalize_prompt_mode(prompt_mode: PromptMode | str | None) -> PromptMode:
    """Return a validated prompt mode, defaulting to the minimal contract."""

    value = "minimal" if prompt_mode is None else str(prompt_mode).strip().lower()
    if value not in PROMPT_MODES:
        raise ValueError(f"prompt_mode must be 'minimal' or 'cooperative', got {prompt_mode!r}")
    return value  # type: ignore[return-value]


# This is intentionally the only complete task/action contract. The
# cooperative mode below is an addendum rather than a second prompt.
_SHARED_TASK_ACTION_CONTRACT = (
    "You are one visitor agent in a shared embodied SimWorld venue-meetup task.\n"
    "The group objective is to identify the single venue feasible for everyone and physically meet there before the shops close.\n\n"
    "The observation for the current turn can contain:\n"
    "- an ego camera image (THIRD-PERSON: your own back is visible, so left/right is unreliable);\n"
    "- coarse-map text/path and self_pose with the current position and heading;\n"
    "- closing_clock with the current simulated time, shop closing time, remaining time, and fixed action cost;\n"
    "- candidate_venues, including each venue id and whether it can be inspected from your area;\n"
    "- known_venue_evidence, readable evidence available in this condition (normally first-hand; the upper bound may synthesize all venue evidence);\n"
    "- private_constraint, requirement information visible in this condition (normally the acting agent's own; full-information may expose all group constraints);\n"
    "- group_chat, messages delivered by other agents;\n"
    "- last_action, last_inspect_result, navigation, landmarks, and valid_actions.\n\n"
    "Actions use the integer in \"choice\":\n"
    "- 0=WAIT (no movement and no message);\n"
    "- 1=STEP_FORWARD with duration and direction (0=forward, 1=backward);\n"
    "- 2=TURN_AROUND with angle in degrees and clockwise;\n"
    "- 3=INSPECT with target_venue_id or target_description;\n"
    "- 4=COMMUNICATE with an optional short message;\n"
    "- 5=NAVIGATE with target_venue_id to travel to its meeting point.\n\n"
    "INSPECT is valid only when the target is permitted by the information partition, "
    "the agent is within the required inspection proximity of the venue, and the target is visible in the "
    "current camera/object-mask view. A successful inspection returns concise, "
    "readable evidence. Do not treat a venue trait as inspected unless that evidence is present.\n"
    "Only choice=4 (COMMUNICATE) sends a message; text or fields attached to any other action are not delivered.\n"
    "Every action consumes the fixed time shown in closing_clock. The shops close when its timer reaches zero, "
    "so inspect, communicate, and converge efficiently.\n"
    "The optional shared_facts field is an evaluator-only annotation for directly inspected "
    "(personally inspected) traits; it is never recipient-visible and is not a parallel communication channel.\n\n"
    "Return exactly one valid JSON object with keys: choice, duration, direction, angle, clockwise, "
    "target_venue_id, target_description, message, shared_facts, reasoning. No markdown or prose outside the JSON object."
)

_COOPERATIVE_ADDENDUM = (
    "Cooperative strategy addendum: disclose your private need to the group, report evidence useful to "
    "teammates, pool observations across agents, and coordinate before convergence on the shared venue."
)


def build_system_prompt(prompt_mode: PromptMode | str = "minimal") -> str:
    """Build the system prompt for the selected mode."""

    mode = normalize_prompt_mode(prompt_mode)
    if mode == "cooperative":
        return f"{_SHARED_TASK_ACTION_CONTRACT}\n\n{_COOPERATIVE_ADDENDUM}"
    return _SHARED_TASK_ACTION_CONTRACT


# Backward-compatible name used by existing callers and archived runs.
VENUE_MEETUP_SYSTEM_PROMPT = build_system_prompt("minimal")


def strip_frame(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an observation copy without image arrays for prompts/logs."""

    return {key: value for key, value in observation.items() if key != "ego_view"}


def build_agent_prompt(
    observation: dict[str, Any],
    prompt_mode: PromptMode | str = "minimal",
) -> str:
    """Build one agent's structured user prompt.

    The system message owns the shared task/action contract.  This user
    message only identifies the condition-specific observation values and
    serializes the current observation, so a cooperative addendum is not
    accidentally delivered twice.
    """

    normalize_prompt_mode(prompt_mode)
    prompt_payload = strip_frame(observation)
    observation_note = (
        "Observation values for this turn are below. known_venue_evidence is readable evidence available in "
        "this condition (first-hand in the main condition; synthesized for the full-information upper bound). "
        "private_constraint is requirement information visible in this condition (normally the acting agent's "
        "own; full-information exposes all group constraints). shared_facts remains an optional evaluator "
        "annotation for personally inspected traits and is not recipient-visible."
    )
    if "known_venue_facts" in prompt_payload:
        observation_note += (
            " All decision facts and all group constraints are intentionally exposed in this full-information "
            "observation, including known_venue_facts."
        )
    return (
        f"{observation_note}\n\nYour observation JSON follows.\n"
        f"{json.dumps(prompt_payload, indent=2, default=str)}\n"
        "Use the system contract to choose one action for this turn."
    )
