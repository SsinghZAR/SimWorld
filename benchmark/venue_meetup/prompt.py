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


def build_targeted_system_prompt(timing_config: dict[str, Any], prompt_mode: PromptMode | str = "minimal") -> str:
    """Mechanics-only contract generated from the same configuration as execution."""

    from benchmark.venue_meetup._core.comms import DEFAULT_MAX_CONTENT_CHARS
    from benchmark.venue_meetup.interactions import action_durations

    contract = (
        "You are one of two visitors in the embodied SimWorld venue-meetup task. "
        "Meet your teammate at the single venue satisfying both visitors' hard requirements by closing. "
        "Your private_constraint states your own hard requirements. Open and unobstructed access are also required. "
        "There are no intermediate rewards, scores, action-specific point penalties, or strategic feedback. "
        "Time spent is the only action cost; evaluation occurs only at the end.\n"
        "The shared simulated clock is independent of model latency and rendering speed. "
        "Each agent can perform one action at a time. Busy agents cannot move, inspect, communicate, "
        "cancel, or start another action; the other visitor acts independently. "
        "Inspection evidence and messages become available only when the corresponding action finishes. "
        "Messages arriving while busy remain in your inbox for your next decision. "
        "Completion exactly at closing is accepted. Unfinished actions do not complete after closing. "
        "The episode ends when both available visitors are at one venue, or at the deadline.\n"
        "Actions: 0=WAIT; 1=STEP_FORWARD (duration in engine seconds, direction 0 forward/1 backward); "
        "2=TURN_AROUND (angle, clockwise); 3=INSPECT (target_interactable_id); "
        "4=COMMUNICATE (message); 5=NAVIGATE (target_venue_id). "
        "Only choice=4 delivers a message. Other choices cannot also communicate. "
        f"Messages are limited to {DEFAULT_MAX_CONTENT_CHARS} characters.\n"
        "NAVIGATE follows a walkable route; its duration estimate appears in navigation.targets. "
        "INSPECT requires a nearby, permitted, currently visible information point from nearby_interactables. "
        "Specify its exact interaction_id in target_interactable_id. Whole-building inspection is not available. "
        "Each source reveals only its own information; unchecked information is unknown, not false. "
        "Information panels describe venue conditions; their appearance does not encode the answer. "
        "Repeated checks give the same evidence. Evidence accumulates in known_venue_evidence and inspection_history. "
        "The ego image is third-person: your own back is visible. self_pose and navigation describe your heading.\n"
        "Partner requirements and evidence are private unless communicated (except the labelled full-information control). "
        "group_chat contains delivered messages. own_activity concerns only you. "
        "shared_facts is optional evaluator-only annotation of directly inspected traits and is never delivered. "
        "Return exactly one JSON object using choice, target_venue_id, target_interactable_id, "
        "duration, direction, angle, clockwise, message, shared_facts, reasoning as needed. No markdown.\n"
        f"Timing configuration: {json.dumps(timing_config, sort_keys=True)}\n"
        f"Action durations in ticks: {json.dumps(action_durations(), sort_keys=True)}"
    )
    if normalize_prompt_mode(prompt_mode) == "cooperative":
        contract += "\n" + _COOPERATIVE_ADDENDUM
    return contract


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
