"""Pure observation assembly for the Venue Meetup environment.

This module contains all logic for building per-agent observation dicts from
pre-collected environment state.  It has no dependency on Unreal Engine or the
Communicator, making it testable offline.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any, Callable, Protocol

from benchmark.venue_meetup.scenario import Scenario, Venue

if TYPE_CHECKING:
    from simworld.utils.vector import Vector


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


def build_observations(
    *,
    scenario: Scenario,
    agent_ids: list[str],
    step_index: int,
    frames: dict[str, Any],
    kinematic_states: dict[str, tuple[_Vec2, float]],
    inboxes: dict[str, list[Any]],
    last_actions: dict[str, dict[str, Any]],
    last_inspections: dict[str, dict[str, Any]],
    revealed_facts: dict[str, dict[str, dict[str, Any]]],
    agent_zone: dict[str, str | None],
    no_coarse_map: bool,
    full_shared_information: bool,
    shared_constraints: bool,
    info_partition: str,
    navigate_mode: str,
    venue_facts_fn: Callable[[Venue], dict[str, Any]],
) -> dict[str, dict[str, Any]]:
    """Assemble per-agent observation dicts from pre-collected state.

    Parameters
    ----------
    kinematic_states:
        Mapping of agent_id to (position_vector, yaw_degrees).
    venue_facts_fn:
        Callable(venue) -> dict returning decision-relevant facts for a venue.
    """

    shared_constraint = "; ".join(agent.private_constraint for agent in scenario.agents)
    observations: dict[str, dict[str, Any]] = {}
    for agent in scenario.agents:
        private_constraint = shared_constraint if shared_constraints else agent.private_constraint
        if full_shared_information:
            venue_summaries: list[Any] = [
                venue.compact() if hasattr(venue, "compact") else venue.__dict__ for venue in scenario.venues
            ]
            known_venue_facts: dict[str, Any] = {venue.venue_id: venue_facts_fn(venue) for venue in scenario.venues}
        else:
            venue_summaries = []
            for venue in scenario.venues:
                summary: dict[str, Any] = {
                    "venue_id": venue.venue_id,
                    "venue_type": venue.venue_type,
                    "slot_id": venue.slot_id,
                    "visual_summary": venue.visual_summary,
                }
                if info_partition == "spatial":
                    summary["zone_id"] = venue.zone_id
                    summary["can_inspect"] = can_inspect_zone(
                        agent.agent_id, venue, info_partition=info_partition, agent_zone=agent_zone,
                    )
                venue_summaries.append(summary)
            known_venue_facts = dict(revealed_facts.get(agent.agent_id, {}))

        position, yaw = kinematic_states[agent.agent_id]
        self_pose, navigation = heading_cue(
            position, yaw, scenario, no_coarse_map=no_coarse_map, navigate_mode=navigate_mode,
        )
        observations[agent.agent_id] = {
            "agent_id": agent.agent_id,
            "step": step_index,
            "max_steps": scenario.max_steps,
            "role": "visitor",
            "objective": "Find the best feasible venue for everyone and physically meet there.",
            "private_constraint": private_constraint,
            "zone_id": agent_zone.get(agent.agent_id),
            "info_partition": info_partition,
            "coarse_map_text": None if no_coarse_map else scenario.coarse_map_text,
            "coarse_map_path": None if no_coarse_map else scenario.coarse_map_path,
            "self_pose": self_pose,
            "candidate_venues": venue_summaries,
            "known_venue_facts": known_venue_facts,
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
            "last_action": last_actions.get(agent.agent_id),
            "last_inspect_result": last_inspections.get(agent.agent_id),
            "valid_actions": ACTION_LEGEND,
            "ego_view": frames[agent.agent_id],
        }
        if navigation is not None:
            observations[agent.agent_id]["navigation"] = navigation
    return observations


def observation_summary(observation: dict[str, Any]) -> dict[str, Any]:
    """Return an image-free observation suitable for compact logs."""

    return {key: value for key, value in observation.items() if key != "ego_view"}
