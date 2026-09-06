"""Typed, reusable information sources and deterministic frontage placement."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

from benchmark.venue_meetup.scenario import Scenario


@dataclass(frozen=True)
class InteractionKind:
    key: str
    label: str
    traits: tuple[str, ...]
    ticks: int


INTERACTION_KINDS = (
    InteractionKind("hours", "Opening-hours notice", ("open",), 1),
    InteractionKind("access", "Entrance/access information", ("accessible", "reachable"), 2),
    InteractionKind("services", "Menu/services board", ("food_drink",), 2),
    InteractionKind("meeting_area", "Meeting-area information", ("capacity", "shelter", "quiet", "uncrowded"), 3),
)
KINDS_BY_KEY = {kind.key: kind for kind in INTERACTION_KINDS}
INTERACTION_RANGE_CM = 1200.0
INTERACTION_MIN_PIXELS = 20


@dataclass(frozen=True)
class InteractionPoint:
    interaction_id: str
    venue_id: str
    kind: InteractionKind
    position: tuple[float, float, float]
    yaw_deg: float
    mask_color: tuple[int, int, int]
    scale: tuple[float, float, float]

    @property
    def actor_name(self) -> str:
        return f"GEN_BP_INTERACTION_{self.interaction_id}"

    def public(self) -> dict[str, Any]:
        return {
            "interaction_id": self.interaction_id,
            "venue_id": self.venue_id,
            "label": self.kind.label,
            "kind": self.kind.key,
            "duration_ticks": self.kind.ticks,
        }


def interaction_points(scenario: Scenario) -> dict[str, InteractionPoint]:
    """Place four neutral information panels outside each facade, never in a route.

    All venues use the same source types and sizes, irrespective of hidden truth.
    Panels report access/amenity conditions; they are not fake ramps or seating.
    """

    points = {}
    for index, venue in enumerate(scenario.venues):
        cx, cy = venue.region.center
        dx, dy = venue.position[0] - cx, venue.position[1] - cy
        length = math.hypot(dx, dy)
        nx, ny = (dx / length, dy / length) if length else (1.0, 0.0)
        for offset, kind in enumerate(INTERACTION_KINDS):
            tangent = (offset - 1.5) * 160
            ident = f"{venue.venue_id}__{kind.key}"
            color_id = index * len(INTERACTION_KINDS) + offset
            points[ident] = InteractionPoint(
                ident, venue.venue_id, kind,
                (round(cx + nx * 350 - ny * tangent, 2), round(cy + ny * 350 + nx * tangent, 2), 190.0),
                math.degrees(math.atan2(ny, nx)) + 90,
                (22 + 25 * (color_id % 8), 35 + 25 * ((color_id // 8) % 8), 45 + 25 * (color_id // 64)),
                (1.0, (1.0, 1.15, 1.3, 1.45)[offset], 0.4),
            )
    return points


def action_durations() -> dict[str, Any]:
    """Shared by the system prompt and observations; not separately hard-coded."""

    return {
        "WAIT": 1, "COMMUNICATE": 1, "STEP_FORWARD": 1, "TURN_AROUND": 1,
        "INSPECT": {kind.key: kind.ticks for kind in INTERACTION_KINDS},
        "NAVIGATE": "ceil(walkable route metres / travel_metres_per_tick), minimum 1",
        "INVALID_ATTEMPT": 1,
    }
