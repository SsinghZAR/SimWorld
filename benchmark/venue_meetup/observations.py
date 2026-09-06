"""Pure observation assembly for the Venue Meetup environment.

This module contains all logic for building per-agent observation dicts from
pre-collected environment state.  It has no dependency on Unreal Engine or the
Communicator, making it testable offline.
"""

from __future__ import annotations

import math
from copy import deepcopy
from typing import Any, Callable, Mapping, Protocol

from benchmark.venue_meetup.closing_clock import ClosingClock
from benchmark.venue_meetup.inspection_evidence import build_inspection_evidence
from benchmark.venue_meetup.scenario import Scenario, Venue


class _Vec2(Protocol):
    x: float
    y: float


ACTION_LEGEND: dict[str, str] = {
    "0": "WAIT",
    "1": "STEP_FORWARD",
    "2": "TURN_AROUND",
    "3": "INSPECT target_venue_id (must be standing in that venue's region - NAVIGATE there first)",
    "4": "COMMUNICATE message",
    "5": "NAVIGATE target_venue_id (walk to a venue's meeting point)",
}


# Ordered allow-lists keep observation serialization deterministic.  Internal
# diagnostics/canonical facts are intentionally absent.
_PUBLIC_TURN_KEYS: tuple[str, ...] = (
    "choice",
    "duration",
    "direction",
    "angle",
    "clockwise",
    "target_venue_id",
    "target_interactable_id",
    "target_description",
    "message",
    "reasoning",
)
_PUBLIC_RESULT_KEYS: tuple[str, ...] = (
    "turn",
    "result",
    "venue_id",
    "interaction_id",
    "started_tick",
    "completed_tick",
    "duration_ticks",
    "target_description",
    "message",
    "agent_visible_result",
    "evidence",
    "reason",
    "arrived",
)


def public_action_result(result: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Return the safe subset of an evaluator action result.

    This allow-list is applied even when a caller accidentally passes an
    internal record directly.  Malformed/non-list evidence is skipped rather
    than copied into an agent observation.
    """

    if result is None:
        return None
    if not isinstance(result, Mapping):
        return {"result": str(result)}
    public: dict[str, Any] = {}
    for key in _PUBLIC_RESULT_KEYS:
        if key not in result:
            continue
        value = result[key]
        if key == "turn":
            if isinstance(value, Mapping):
                public[key] = {
                    turn_key: deepcopy(value[turn_key])
                    for turn_key in _PUBLIC_TURN_KEYS
                    if turn_key in value
                }
            continue
        if key == "evidence":
            if isinstance(value, list):
                public[key] = [sentence for sentence in value if isinstance(sentence, str)]
            continue
        public[key] = deepcopy(value)
    return public


def normalize_angle(angle: float) -> float:
    """Normalize an angle to [-180, 180]."""

    while angle > 180:
        angle -= 360
    while angle < -180:
        angle += 360
    return angle


_COMPASS_POINTS = ("east", "north-east", "north", "north-west", "west", "south-west", "south", "south-east")


def compass_label(angle_deg: float) -> str:
    """Map a world angle (0=east/+x, 90=north/+y, CCW) to an 8-point compass label."""

    index = int((angle_deg % 360 + 22.5) // 45) % 8
    return _COMPASS_POINTS[index]


def turn_to_face(heading_deg: float, bearing_deg: float, *, tolerance: float = 8.0) -> dict[str, Any]:
    """Describe the TURN_AROUND that aligns ``heading_deg`` onto ``bearing_deg``.

    Uses the env's movement convention: ``clockwise=False`` increases yaw (a
    counter-clockwise / left turn on the north-up coarse map), ``clockwise=True``
    decreases it. Following this then STEP_FORWARD is guaranteed to approach the
    target because forward motion is along ``(cos yaw, sin yaw)``.
    """

    delta = normalize_angle(bearing_deg - heading_deg)
    if abs(delta) <= tolerance:
        return {"instruction": "already facing it (within ~8 deg) - STEP_FORWARD to approach", "needs_turn": False}
    if delta > 0:
        return {
            "instruction": f"turn LEFT ~{round(delta)} deg, then STEP_FORWARD",
            "needs_turn": True,
            "action": {"choice": 2, "clockwise": False, "angle": round(delta)},
        }
    return {
        "instruction": f"turn RIGHT ~{round(-delta)} deg, then STEP_FORWARD",
        "needs_turn": True,
        "action": {"choice": 2, "clockwise": True, "angle": round(-delta)},
    }


def vector_to_dict(vector: _Vec2) -> dict[str, float]:
    """Serialize a Vector."""

    return {"x": float(vector.x), "y": float(vector.y)}


def target_cue(
    ident: str,
    kind: str,
    type_: str,
    target_pos: Any,
    agent_pos: _Vec2,
    yaw: float,
    region: Any | None = None,
) -> dict[str, Any]:
    """Build a world-frame bearing/turn/distance cue toward one target."""

    dx = float(target_pos[0]) - agent_pos.x
    dy = float(target_pos[1]) - agent_pos.y
    bearing = math.degrees(math.atan2(dy, dx))
    turn = turn_to_face(yaw, bearing)
    arrived = bool(region is not None and region.contains((agent_pos.x, agent_pos.y)))
    cue: dict[str, Any] = {
        "id": ident,
        "kind": kind,
        "type": type_,
        "direction": compass_label(bearing),
        "bearing_deg": round(bearing),
        "distance_m": round(math.hypot(dx, dy) / 100.0),
        "guidance": "You are here (at this venue). Stop advancing; INSPECT or WAIT." if arrived else turn["instruction"],
    }
    if arrived:
        cue["arrived"] = True
    elif turn.get("action"):
        cue["suggested_action"] = turn["action"]
    return cue


def heading_cue(
    agent_pos: _Vec2,
    yaw: float,
    scenario: Scenario,
    *,
    no_coarse_map: bool,
    navigate_mode: str,
) -> tuple[dict[str, Any], dict[str, Any] | None]:
    """Compute proprioceptive heading and (map-gated) navigation cues."""

    self_pose = {
        "facing": compass_label(yaw),
        "heading_deg": round(yaw),
        "note": (
            "Your camera is third-person (you see your own back). Use this compass and the coarse map "
            "(north=up/+y, east=right/+x) for left/right decisions, not the image."
        ),
    }
    if no_coarse_map:
        return self_pose, None
    targets = [
        target_cue(venue.venue_id, "venue", venue.venue_type, venue.position, agent_pos, yaw, region=venue.region)
        for venue in scenario.venues
    ]
    targets += [
        target_cue(landmark.landmark_id, "landmark", landmark.landmark_type, landmark.position, agent_pos, yaw)
        for landmark in scenario.landmarks
    ]
    if navigate_mode == "walk":
        hint = (
            "Use NAVIGATE (choice=5, target_venue_id) to walk to a venue: it plans a route around the buildings and "
            "physically walks you toward that venue's meeting region (this takes real travel time and can be blocked "
            "by a building - if 'arrived' is false, NAVIGATE again to keep going). You must be in a venue's region to "
            "INSPECT it. STEP_FORWARD/TURN_AROUND are optional fine movement; venues are solid buildings you cannot "
            "walk through. When a venue shows 'arrived: true' you are physically at it - INSPECT it or WAIT."
        )
    else:
        hint = (
            "Prefer NAVIGATE (choice=5, target_venue_id) to travel to a venue in one action: it places you in that "
            "venue's meeting region. You must be in a venue's region to INSPECT it. STEP_FORWARD/TURN_AROUND are "
            "optional fine movement; venues are solid buildings you cannot walk through. When a venue shows "
            "'arrived: true' you are physically at it - INSPECT it or WAIT."
        )
    navigation = {
        "frame": "world bearings (north=up/+y, east=right/+x); matches the coarse map",
        "hint": hint,
        "targets": targets,
    }
    return self_pose, navigation


def can_inspect_zone(agent_id: str, venue: Venue, *, info_partition: str, agent_zone: dict[str, str | None]) -> bool:
    """Return whether the partition mode lets this agent inspect this venue."""

    if info_partition != "spatial":
        return True
    zone = agent_zone.get(agent_id)
    if venue.zone_id is None or zone is None:
        return True
    return venue.zone_id == zone


def _candidate_summary(
    agent_id: str,
    venue: Venue,
    *,
    info_partition: str,
    agent_zone: Mapping[str, str | None],
) -> dict[str, Any]:
    """Return the public identity/navigation summary for a venue.

    Do not serialize ``Venue`` wholesale: its properties, region, asset path,
    and mask colour are evaluator/runtime metadata.  This explicit allow-list
    is shared by normal and full-information observations.
    """

    summary: dict[str, Any] = {
        "venue_id": venue.venue_id,
        "venue_type": venue.venue_type,
        "slot_id": venue.slot_id,
        "visual_summary": venue.visual_summary,
    }
    if info_partition == "spatial":
        summary["zone_id"] = venue.zone_id
        summary["can_inspect"] = can_inspect_zone(
            agent_id,
            venue,
            info_partition=info_partition,
            agent_zone=dict(agent_zone),
        )
    return summary


def build_observations(
    *,
    scenario: Scenario,
    agent_ids: list[str],
    step_index: int,
    frames: dict[str, Any],
    kinematic_states: dict[str, tuple[_Vec2, float]],
    inboxes: Mapping[str, list[Any]],
    last_actions: Mapping[str, dict[str, Any]] | None = None,
    last_inspections: Mapping[str, dict[str, Any]] | None = None,
    last_inspections_public: Mapping[str, dict[str, Any]] | None = None,
    revealed_facts: Mapping[str, Mapping[str, dict[str, Any]]] | None = None,
    revealed_evidence: Mapping[str, Mapping[str, list[str]]] | None = None,
    agent_zone: Mapping[str, str | None] | None = None,
    no_coarse_map: bool = False,
    full_shared_information: bool = False,
    shared_constraints: bool = False,
    info_partition: str = "none",
    navigate_mode: str = "teleport",
    closing_clock: Mapping[str, Any] | None = None,
    venue_facts_fn: Callable[[Venue], dict[str, Any]] | None = None,
) -> dict[str, dict[str, Any]]:
    """Assemble per-agent observation dicts from pre-collected state.

    Parameters
    ----------
    kinematic_states:
        Mapping of agent_id to (position_vector, yaw_degrees).
    venue_facts_fn:
        Callable(venue) -> dict returning decision-relevant facts for a venue.

    ``known_venue_evidence`` contains only ordered readable sentences derived
    from successful first-hand inspections.  ``known_venue_facts`` is emitted
    only for the explicit ``full_shared_information`` upper-bound ablation.
    """

    agent_zone = agent_zone or {}
    revealed_facts = revealed_facts or {}
    revealed_evidence = revealed_evidence or {}
    last_actions = last_actions or {}
    clock_state = (
        deepcopy(closing_clock)
        if closing_clock is not None
        else ClosingClock(scenario.max_steps).snapshot(step_index)
    )
    # ``last_inspections`` is the old parameter name.  New callers must pass
    # the explicitly public store; retaining the fallback keeps offline tools
    # that assemble observations directly source-compatible.
    if last_inspections_public is None:
        last_inspections_public = last_inspections or {}
    venue_facts_fn = venue_facts_fn or (lambda venue: {})

    shared_constraint = "; ".join(agent.private_constraint for agent in scenario.agents)
    observations: dict[str, dict[str, Any]] = {}
    for agent in scenario.agents:
        private_constraint = shared_constraint if shared_constraints else agent.private_constraint
        venue_summaries = [
            _candidate_summary(agent.agent_id, venue, info_partition=info_partition, agent_zone=agent_zone)
            for venue in scenario.venues
        ]
        if full_shared_information:
            # The upper-bound ablation intentionally exposes canonical facts to
            # every agent, but candidate metadata remains the safe identity
            # summary above.
            known_venue_facts: dict[str, Any] = {
                venue.venue_id: deepcopy(venue_facts_fn(venue)) for venue in scenario.venues
            }
            known_venue_evidence: dict[str, list[str]] = {
                venue_id: list(build_inspection_evidence(facts, venue_id=venue_id).public_evidence)
                for venue_id, facts in known_venue_facts.items()
            }
        else:
            # Canonical facts remain evaluator-only in the main observation.
            known_venue_evidence = {
                venue_id: list(sentences)
                for venue_id, sentences in revealed_evidence.get(agent.agent_id, {}).items()
            }
            # Offline callers from the pre-evidence API may still provide only
            # ``revealed_facts``.  Derive readable evidence locally without
            # placing those canonical dictionaries in the returned observation.
            if not known_venue_evidence and revealed_facts.get(agent.agent_id):
                known_venue_evidence = {
                    venue_id: list(build_inspection_evidence(facts, venue_id=venue_id).public_evidence)
                    for venue_id, facts in revealed_facts[agent.agent_id].items()
                }

        position, yaw = kinematic_states[agent.agent_id]
        self_pose, navigation = heading_cue(
            position, yaw, scenario, no_coarse_map=no_coarse_map, navigate_mode=navigate_mode,
        )
        agent_observation: dict[str, Any] = {
            "agent_id": agent.agent_id,
            "step": step_index,
            "max_steps": scenario.max_steps,
            "closing_clock": deepcopy(clock_state),
            "role": "visitor",
            "objective": "Find the best feasible venue for everyone and physically meet there before the shops close.",
            "private_constraint": private_constraint,
            "zone_id": agent_zone.get(agent.agent_id),
            "info_partition": info_partition,
            "coarse_map_text": None if no_coarse_map else scenario.coarse_map_text,
            "coarse_map_path": None if no_coarse_map else scenario.coarse_map_path,
            "self_pose": self_pose,
            "candidate_venues": venue_summaries,
            "known_venue_evidence": known_venue_evidence,
            "landmarks": [
                {
                    "landmark_id": landmark.landmark_id,
                    "type": landmark.landmark_type,
                    "slot_id": landmark.slot_id,
                    "visual_summary": landmark.visual_summary,
                }
                for landmark in scenario.landmarks
            ],
            "group_chat": [message.compact_for_recipient() for message in inboxes.get(agent.agent_id, [])],
            "roster": agent_ids,
            "last_action": public_action_result(last_actions.get(agent.agent_id)),
            "last_inspect_result": public_action_result(last_inspections_public.get(agent.agent_id)),
            "valid_actions": ACTION_LEGEND,
            "ego_view": frames[agent.agent_id],
        }
        if full_shared_information:
            agent_observation["known_venue_facts"] = known_venue_facts
        if navigation is not None:
            agent_observation["navigation"] = navigation
        observations[agent.agent_id] = agent_observation
    return observations


def observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an image-free observation suitable for compact logs."""

    return {key: value for key, value in observation.items() if key != "ego_view"}
