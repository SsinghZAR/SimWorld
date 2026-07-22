"""Single-action dispatch and inspection handling for Venue Meetup."""

from __future__ import annotations

from typing import Any, Callable

import numpy as np

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn
from benchmark.venue_meetup.observations import can_inspect_zone
from benchmark.venue_meetup.scenario import Venue
from simworld.utils.vector import Vector


def resolve_inspect_target(
    venues: list[Venue],
    agent_position: tuple[float, float],
    target_venue_id: str | None = None,
    target_description: str | None = None,
) -> Venue | None:
    """Resolve an inspection/navigate target by id, description, or proximity.

    Priority: exact venue_id match > substring description match > nearest venue.
    Returns None only when *venues* is empty.
    """

    if target_venue_id:
        for venue in venues:
            if venue.venue_id == target_venue_id:
                return venue
    if target_description:
        query = target_description.lower()
        for venue in venues:
            if query in venue.venue_id.lower() or query in venue.visual_summary.lower() or query in venue.venue_type:
                return venue
    if not venues:
        return None
    pos = Vector(agent_position[0], agent_position[1])
    return min(venues, key=lambda v: pos.distance(Vector(v.region.center[0], v.region.center[1])))


def compute_inspection(
    venue: Venue,
    agent_id: str,
    action: VenueAgentTurn,
    *,
    agent_xy: tuple[float, float],
    mask_pixels: int,
    info_partition: str,
    agent_zone: dict[str, str | None],
    venue_facts_fn: Callable[[Venue], dict[str, Any]],
) -> tuple[dict[str, Any], bool]:
    """Compute the inspection result for a resolved venue target.

    Returns ``(result_dict, valid)`` where *valid* indicates a successful
    inspection that should update bookkeeping (inspected set, revealed facts).
    """

    if not can_inspect_zone(agent_id, venue, info_partition=info_partition, agent_zone=agent_zone):
        return {
            "result": "INSPECT_FAILED",
            "venue_id": venue.venue_id,
            "reason": "outside your area",
            "agent_visible_result": "this venue is in your teammate's area; ask them to inspect it and report back",
        }, False

    at_venue = venue.region.contains(agent_xy)
    distance_to_center = Vector(agent_xy[0], agent_xy[1]).distance(
        Vector(venue.region.center[0], venue.region.center[1])
    )
    valid = at_venue
    result: dict[str, Any] = {
        "result": "INSPECT_OK" if valid else "INSPECT_FAILED",
        "venue_id": venue.venue_id,
        "target_description": action.target_description,
        "distance_to_center_internal": round(distance_to_center, 2),
        "region_radius_internal": round(float(venue.region.radius), 2),
        "mask_pixels_internal": int(mask_pixels),
        "agent_visible_result": (
            "focused camera frame returned"
            if valid
            else "you are not at this venue yet - NAVIGATE to it first, then INSPECT"
        ),
    }
    if valid:
        result["facts"] = venue_facts_fn(venue)
    return result, valid


def count_mask_pixels(frame: Any, color_rgb: tuple[int, int, int]) -> int:
    """Count approximate RGB or BGR venue-color pixels in an object-mask frame."""

    if frame is None or not hasattr(frame, "shape") or len(frame.shape) < 3:
        return 0
    rgb = np.array(color_rgb, dtype=np.int16)
    bgr = np.array((color_rgb[2], color_rgb[1], color_rgb[0]), dtype=np.int16)
    pixels = frame[:, :, :3].astype(np.int16)
    rgb_hits = np.all(np.abs(pixels - rgb) <= 8, axis=2)
    bgr_hits = np.all(np.abs(pixels - bgr) <= 8, axis=2)
    return int(np.count_nonzero(rgb_hits | bgr_hits))


def dispatch_single_action(
    agent_id: str,
    action: VenueAgentTurn,
    *,
    step_forward_fn: Callable[[str, VenueAgentTurn], dict[str, Any]],
    rotate_fn: Callable[[str, float, str], dict[str, Any]],
    inspect_fn: Callable[[str, VenueAgentTurn], dict[str, Any]],
    navigate_fn: Callable[[str, VenueAgentTurn], dict[str, Any]],
    stop_fn: Callable[[str], None],
) -> dict[str, Any]:
    """Route a single-agent action to the appropriate handler.

    *step_forward_fn* must return a complete result dict (including the
    ``"turn"`` key).  All other handlers return a partial dict that is
    merged with ``{"turn": action.compact()}``.
    """

    if action.choice == VenueAction.STEP_FORWARD.value:
        return step_forward_fn(agent_id, action)
    if action.choice == VenueAction.TURN_AROUND.value:
        result = rotate_fn(agent_id, float(action.angle or 45), "right" if action.clockwise else "left")
    elif action.choice == VenueAction.INSPECT.value:
        result = inspect_fn(agent_id, action)
    elif action.choice == VenueAction.NAVIGATE.value:
        result = navigate_fn(agent_id, action)
    elif action.choice == VenueAction.COMMUNICATE.value:
        result = {"result": "COMMUNICATE", "message": action.message}
    else:
        stop_fn(agent_id)
        result = {"result": "WAIT"}
    return {"turn": action.compact(), **result}
