"""Single-action dispatch and inspection handling for Venue Meetup.

Inspection has two intentionally separate outputs.  The evaluator receives a
canonical first-hand record (including diagnostics and facts on success),
while agents receive a small public record with deterministic readable
evidence.  Keeping the split here makes it difficult for observation assembly
to accidentally leak hidden state.
"""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
from typing import Any, Callable, Mapping, TypedDict

import numpy as np

from benchmark.venue_meetup._core.action_space import VenueAction, VenueAgentTurn
from benchmark.venue_meetup.inspection_evidence import InspectionEvidence, build_inspection_evidence
from benchmark.venue_meetup.observations import can_inspect_zone
from benchmark.venue_meetup.scenario import Venue
from simworld.utils.vector import Vector


class InspectionInternalRecord(TypedDict, total=False):
    """Evaluator-only inspection fields.

    ``TypedDict`` documents the wire shape while retaining JSON-compatible
    dictionaries for existing trajectory and social-metric consumers.
    """

    result: str
    venue_id: str
    target_description: str | None
    distance_to_center_internal: float
    region_radius_internal: float
    inspect_range_internal: float | None
    in_region_internal: bool
    in_range_internal: bool
    mask_pixels_internal: int
    inspect_min_mask_pixels_internal: int
    facts: dict[str, Any]
    evidence_by_trait_internal: dict[str, str]
    reason: str
    agent_visible_result: str


class InspectionPublicRecord(TypedDict, total=False):
    """Agent-visible inspection fields; no canonical values or diagnostics."""

    result: str
    venue_id: str
    target_description: str | None
    reason: str
    agent_visible_result: str
    evidence: list[str]


@dataclass(frozen=True)
class InspectionOutcome:
    """Result of an inspection attempt with evaluator/public records split."""

    success: bool
    internal_record: InspectionInternalRecord
    public_record: InspectionPublicRecord


@dataclass(frozen=True)
class InspectionPrecheck:
    """Pure permission/proximity gate executed before camera-mask capture."""

    success: bool
    internal_record: InspectionInternalRecord
    public_record: InspectionPublicRecord


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


def _distance_to_region_center(venue: Venue, agent_xy: tuple[float, float]) -> float:
    """Return the 2D distance used by the inspect-range gate."""

    return Vector(agent_xy[0], agent_xy[1]).distance(Vector(venue.region.center[0], venue.region.center[1]))


def _precheck_records(
    venue: Venue,
    action: VenueAgentTurn,
    *,
    distance_to_center: float,
    inspect_range: float | None,
    in_region: bool,
    in_range: bool,
    success: bool,
    reason: str,
    visible_result: str,
) -> InspectionPrecheck:
    """Build typed precheck records while keeping diagnostics evaluator-only."""

    internal: InspectionInternalRecord = {
        "result": "INSPECT_OK" if success else "INSPECT_FAILED",
        "venue_id": venue.venue_id,
        "target_description": action.target_description,
        "distance_to_center_internal": round(distance_to_center, 2),
        "region_radius_internal": round(float(venue.region.radius), 2),
        "inspect_range_internal": None if inspect_range is None else round(float(inspect_range), 2),
        "in_region_internal": in_region,
        "in_range_internal": in_range,
        # Kept in the evaluator record for the legacy tuple-returning helper;
        # public observations still use the separately copied public record.
        "agent_visible_result": visible_result,
    }
    public: InspectionPublicRecord = {
        "result": "INSPECT_OK" if success else "INSPECT_FAILED",
        "venue_id": venue.venue_id,
        "target_description": action.target_description,
        "reason": reason,
        "agent_visible_result": visible_result,
    }
    if reason:
        internal["reason"] = reason
    return InspectionPrecheck(success=success, internal_record=internal, public_record=public)


def precheck_inspection(
    venue: Venue,
    agent_id: str,
    action: VenueAgentTurn,
    *,
    agent_xy: tuple[float, float],
    info_partition: str = "none",
    agent_zone: Mapping[str, str | None] | None = None,
    inspect_range: float | None = None,
) -> InspectionPrecheck:
    """Check inspect permission and proximity without touching the camera.

    The order is deliberate: a spatially disallowed or non-proximate target
    cannot trigger an object-mask capture.  ``inspect_range=None`` preserves
    legacy behavior while still requiring membership in ``venue.region``.
    """

    if not can_inspect_zone(agent_id, venue, info_partition=info_partition, agent_zone=dict(agent_zone or {})):
        distance = _distance_to_region_center(venue, agent_xy)
        in_region = bool(venue.region.contains(agent_xy))
        in_range = inspect_range is None or distance <= float(inspect_range)
        return _precheck_records(
            venue,
            action,
            distance_to_center=distance,
            inspect_range=inspect_range,
            in_region=in_region,
            in_range=bool(in_range),
            success=False,
            reason="outside your area",
            visible_result="this venue is in your teammate's area; ask them to inspect it and report back",
        )

    in_region = bool(venue.region.contains(agent_xy))
    distance = _distance_to_region_center(venue, agent_xy)
    in_range = inspect_range is None or distance <= float(inspect_range)
    if not in_region:
        return _precheck_records(
            venue,
            action,
            distance_to_center=distance,
            inspect_range=inspect_range,
            in_region=False,
            in_range=bool(in_range),
            success=False,
            reason="not at this venue",
            visible_result="you are not at this venue yet - NAVIGATE to it first, then INSPECT",
        )
    if not in_range:
        return _precheck_records(
            venue,
            action,
            distance_to_center=distance,
            inspect_range=inspect_range,
            in_region=True,
            in_range=False,
            success=False,
            reason="too far from this venue",
            visible_result="you are too far from this venue to inspect it - move closer and try again",
        )
    return _precheck_records(
        venue,
        action,
        distance_to_center=distance,
        inspect_range=inspect_range,
        in_region=True,
        in_range=True,
        success=True,
        reason="",
        visible_result="the target is in range; a visible object mask will be checked",
    )


def complete_inspection(
    venue: Venue,
    action: VenueAgentTurn,
    precheck: InspectionPrecheck,
    *,
    mask_pixels: int,
    inspect_min_mask_pixels: int,
    venue_facts_fn: Callable[[Venue], dict[str, Any]],
) -> InspectionOutcome:
    """Finish a passing precheck using the current-orientation object mask.

    Facts and readable evidence are generated only after the visibility
    threshold passes.  A failed mask check receives no fact payload.
    """

    internal: InspectionInternalRecord = deepcopy(precheck.internal_record)
    public: InspectionPublicRecord = deepcopy(precheck.public_record)
    if not precheck.success:
        return InspectionOutcome(False, internal, public)

    pixels = int(mask_pixels)
    threshold = max(0, int(inspect_min_mask_pixels))
    internal["mask_pixels_internal"] = pixels
    internal["inspect_min_mask_pixels_internal"] = threshold
    if pixels < threshold:
        internal["result"] = "INSPECT_FAILED"
        internal["reason"] = "target not visible enough"
        internal["agent_visible_result"] = "the venue is not clearly visible from your current view; face it and try again"
        public.update(
            {
                "result": "INSPECT_FAILED",
                "reason": "target not visible enough",
                "agent_visible_result": "the venue is not clearly visible from your current view; face it and try again",
            }
        )
        return InspectionOutcome(False, internal, public)

    facts = deepcopy(dict(venue_facts_fn(venue)))
    evidence: InspectionEvidence = build_inspection_evidence(facts, venue_id=venue.venue_id)
    internal["result"] = "INSPECT_OK"
    internal["agent_visible_result"] = "focused camera frame returned"
    internal["facts"] = facts
    internal["evidence_by_trait_internal"] = evidence.internal_mapping
    public.update(
        {
            "result": "INSPECT_OK",
            "reason": "",
            "agent_visible_result": "focused camera frame returned",
            "evidence": evidence.public_evidence,
        }
    )
    return InspectionOutcome(True, internal, public)


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
    inspect_range: float | None = None,
    inspect_min_mask_pixels: int | None = None,
) -> tuple[dict[str, Any], bool]:
    """Compatibility wrapper returning the historical ``(dict, bool)`` pair.

    New environment code should use :func:`precheck_inspection` followed by
    :func:`complete_inspection` so it can enforce camera-call ordering.  This
    pure wrapper remains useful to older offline callers and tests.
    """

    precheck = precheck_inspection(
        venue,
        agent_id,
        action,
        agent_xy=agent_xy,
        info_partition=info_partition,
        agent_zone=agent_zone,
        inspect_range=inspect_range,
    )
    # Match the environment's default visibility gate for direct pure-helper
    # callers; the live env supplies its configured threshold explicitly.
    threshold = 50 if inspect_min_mask_pixels is None else inspect_min_mask_pixels
    outcome = complete_inspection(
        venue,
        action,
        precheck,
        mask_pixels=mask_pixels,
        inspect_min_mask_pixels=threshold,
        venue_facts_fn=venue_facts_fn,
    )
    # Preserve the old helper's mixed result shape for offline callers.  The
    # live environment uses the typed outcome directly and never exposes this
    # merged dictionary to an agent.
    record: dict[str, Any] = deepcopy(outcome.internal_record)
    record.update(deepcopy(outcome.public_record))
    return record, outcome.success


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
