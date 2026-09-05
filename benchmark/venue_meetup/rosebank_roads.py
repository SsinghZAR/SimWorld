"""Pure visual road-dressing plan for the Rosebank-inspired grid.

The packaged ``BP_Road1`` blueprint is incompatible with the empty map, so the
road layer uses three known-stable catalogue blueprints as thin, non-colliding
slabs. Their measured geometry supplies asphalt, raised pavements, lane paint,
and zebra crossings without changing the authored walk graph or physics.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Literal

from benchmark.venue_meetup.building_catalog import asset_path, building_bbox
from benchmark.venue_meetup.layout import DistrictLayout, StreetSegment
from benchmark.venue_meetup.rosebank_grid import ROSEBANK_GRID_TEMPLATE_IDS

RoadActorKind = Literal["carriageway", "sidewalk", "lane_marking", "crosswalk"]

ROSEBANK_LAYOUT_ID = "rosebank_grid_9x9_v0"
ROSEBANK_LAYOUT_IDS = frozenset(ROSEBANK_GRID_TEMPLATE_IDS.values())
ASPHALT_ASSET_KEY = "BP_Building_44_C"
SIDEWALK_ASSET_KEY = "BP_Building_05_C"
MARKING_ASSET_KEY = "BP_RoadBlocker_C"

_ASPHALT_HEIGHT_SCALE = 0.005
_SIDEWALK_HEIGHT_SCALE = 0.025
_MARKING_HEIGHT_SCALE = 0.04
_ASPHALT_TOP_CM = 2.0
_SIDEWALK_TOP_CM = 14.0
_MARKING_Z_CM = 3.0

# Measured live from the packaged BP_RoadBlocker blueprint. It is absent from
# bounding_boxes.json because that file only profiles building assets.
_MARKING_LENGTH_CM = 165.554
_MARKING_WIDTH_CM = 57.25
_CROSSWALK_BAR_COUNT = 4
_CROSSWALK_BAR_WIDTH_CM = 90.0
_CROSSWALK_BAR_SPACING_CM = 155.0
_DOUBLE_LINE_OFFSET_CM = 45.0


@dataclass(frozen=True, slots=True)
class RosebankRoadActor:
    """One inert blueprint slab used to make an authored street readable."""

    actor_id: str
    kind: RoadActorKind
    street_id: str
    asset_key: str
    asset_path: str
    position: tuple[float, float, float]
    yaw_deg: float
    scale: tuple[float, float, float]
    collision: bool = False
    movable: bool = False


def _is_horizontal(street: StreetSegment) -> bool:
    return math.isclose(street.start[1], street.end[1], abs_tol=1e-6)


def _street_center(street: StreetSegment) -> tuple[float, float]:
    return (
        (street.start[0] + street.end[0]) / 2.0,
        (street.start[1] + street.end[1]) / 2.0,
    )


def _street_length(street: StreetSegment) -> float:
    return math.dist(street.start, street.end)


def _carriageway_width(street: StreetSegment) -> float:
    return max(200.0, street.width_cm - 2.0 * street.sidewalk_width_cm)


def _slab_z(asset_key: str, height_scale: float, top_cm: float) -> float:
    return top_cm - building_bbox(asset_key)[2] * height_scale


def _surface_actor(street: StreetSegment) -> RosebankRoadActor:
    raw_x, raw_y, _raw_z = building_bbox(ASPHALT_ASSET_KEY)
    length = _street_length(street)
    width = _carriageway_width(street)
    horizontal = _is_horizontal(street)
    # BP_Building_44's mesh axes are rotated inside its blueprint: scale.x
    # controls the long world-y direction at yaw 0, and scale.y controls width.
    scale = (
        length / raw_y,
        width / raw_x,
        _ASPHALT_HEIGHT_SCALE,
    )
    center = _street_center(street)
    return RosebankRoadActor(
        actor_id=f"GEN_BP_ROAD_SURFACE_{street.street_id}",
        kind="carriageway",
        street_id=street.street_id,
        asset_key=ASPHALT_ASSET_KEY,
        asset_path=asset_path(ASPHALT_ASSET_KEY),
        position=(
            center[0],
            center[1],
            _slab_z(
                ASPHALT_ASSET_KEY,
                _ASPHALT_HEIGHT_SCALE,
                _ASPHALT_TOP_CM,
            ),
        ),
        yaw_deg=90.0 if horizontal else 0.0,
        scale=scale,
    )


def _sidewalk_actors(street: StreetSegment) -> tuple[RosebankRoadActor, ...]:
    raw_x, raw_y, _raw_z = building_bbox(SIDEWALK_ASSET_KEY)
    length = _street_length(street)
    width = street.sidewalk_width_cm
    horizontal = _is_horizontal(street)
    center = _street_center(street)
    offset = _carriageway_width(street) / 2.0 + width / 2.0
    normal = (0.0, 1.0) if horizontal else (1.0, 0.0)
    scale = (
        length / raw_x,
        width / raw_y,
        _SIDEWALK_HEIGHT_SCALE,
    )
    z = _slab_z(
        SIDEWALK_ASSET_KEY,
        _SIDEWALK_HEIGHT_SCALE,
        _SIDEWALK_TOP_CM,
    )
    return tuple(
        RosebankRoadActor(
            actor_id=f"GEN_BP_ROAD_SIDEWALK_{street.street_id}_{side}",
            kind="sidewalk",
            street_id=street.street_id,
            asset_key=SIDEWALK_ASSET_KEY,
            asset_path=asset_path(SIDEWALK_ASSET_KEY),
            position=(
                center[0] + normal[0] * offset * sign,
                center[1] + normal[1] * offset * sign,
                z,
            ),
            yaw_deg=0.0 if horizontal else 90.0,
            scale=scale,
        )
        for side, sign in (("left", -1.0), ("right", 1.0))
    )


def _marking_actor(
    *,
    actor_id: str,
    kind: RoadActorKind,
    street_id: str,
    center: tuple[float, float],
    length_cm: float,
    width_cm: float,
    horizontal: bool,
) -> RosebankRoadActor:
    return RosebankRoadActor(
        actor_id=actor_id,
        kind=kind,
        street_id=street_id,
        asset_key=MARKING_ASSET_KEY,
        asset_path=asset_path(MARKING_ASSET_KEY),
        position=(center[0], center[1], _MARKING_Z_CM),
        yaw_deg=0.0 if horizontal else 90.0,
        scale=(
            length_cm / _MARKING_LENGTH_CM,
            width_cm / _MARKING_WIDTH_CM,
            _MARKING_HEIGHT_SCALE,
        ),
    )


def _lane_marking_actors(
    street: StreetSegment,
) -> tuple[RosebankRoadActor, ...]:
    primary = street.street_id in {"oxford_road", "tyrwhitt_high_street"}
    if not primary and street.width_cm < 1_500.0:
        return ()
    horizontal = _is_horizontal(street)
    center = _street_center(street)
    offsets = (-_DOUBLE_LINE_OFFSET_CM, _DOUBLE_LINE_OFFSET_CM) if primary else (0.0,)
    normal = (0.0, 1.0) if horizontal else (1.0, 0.0)
    return tuple(
        _marking_actor(
            actor_id=f"GEN_BP_ROAD_LINE_{street.street_id}_{index}",
            kind="lane_marking",
            street_id=street.street_id,
            center=(
                center[0] + normal[0] * offset,
                center[1] + normal[1] * offset,
            ),
            length_cm=_street_length(street) - 300.0,
            width_cm=24.0,
            horizontal=horizontal,
        )
        for index, offset in enumerate(offsets)
    )


def _crosswalk_bar_actors(
    *,
    prefix: str,
    street_id: str,
    intersection: tuple[float, float],
    crossing_length_cm: float,
    horizontal: bool,
) -> tuple[RosebankRoadActor, ...]:
    tangent = (0.0, 1.0) if horizontal else (1.0, 0.0)
    center_index = (_CROSSWALK_BAR_COUNT - 1) / 2.0
    return tuple(
        _marking_actor(
            actor_id=f"GEN_BP_ROAD_CROSSWALK_{prefix}_{bar_index}",
            kind="crosswalk",
            street_id=street_id,
            center=(
                intersection[0]
                + tangent[0]
                * (bar_index - center_index)
                * _CROSSWALK_BAR_SPACING_CM,
                intersection[1]
                + tangent[1]
                * (bar_index - center_index)
                * _CROSSWALK_BAR_SPACING_CM,
            ),
            length_cm=max(120.0, crossing_length_cm - 80.0),
            width_cm=_CROSSWALK_BAR_WIDTH_CM,
            horizontal=horizontal,
        )
        for bar_index in range(_CROSSWALK_BAR_COUNT)
    )


def _crosswalk_actors(
    streets: tuple[StreetSegment, ...],
) -> tuple[RosebankRoadActor, ...]:
    vertical = tuple(street for street in streets if not _is_horizontal(street))
    horizontal = tuple(street for street in streets if _is_horizontal(street))
    oxford = next(street for street in vertical if street.street_id == "oxford_road")
    tyrwhitt = next(
        street for street in horizontal if street.street_id == "tyrwhitt_high_street"
    )
    actors: list[RosebankRoadActor] = []
    for index, cross_street in enumerate(horizontal):
        actors.extend(
            _crosswalk_bar_actors(
                prefix=f"OXFORD_{index}",
                street_id=oxford.street_id,
                intersection=(oxford.start[0], cross_street.start[1]),
                crossing_length_cm=_carriageway_width(oxford),
                horizontal=True,
            )
        )
    for index, cross_street in enumerate(vertical):
        actors.extend(
            _crosswalk_bar_actors(
                prefix=f"TYRWHITT_{index}",
                street_id=tyrwhitt.street_id,
                intersection=(cross_street.start[0], tyrwhitt.start[1]),
                crossing_length_cm=_carriageway_width(tyrwhitt),
                horizontal=False,
            )
        )
    return tuple(actors)


def plan_rosebank_road_actors(
    layout: DistrictLayout | None,
) -> tuple[RosebankRoadActor, ...]:
    """Return inert road actors for supported grids and nothing for other maps."""

    if layout is None or layout.layout_id not in ROSEBANK_LAYOUT_IDS:
        return ()
    streets = tuple(layout.streets)
    actors: list[RosebankRoadActor] = []
    for street in streets:
        actors.append(_surface_actor(street))
        actors.extend(_sidewalk_actors(street))
        actors.extend(_lane_marking_actors(street))
    actors.extend(_crosswalk_actors(streets))
    return tuple(actors)


__all__ = [
    "ASPHALT_ASSET_KEY",
    "MARKING_ASSET_KEY",
    "ROSEBANK_LAYOUT_ID",
    "ROSEBANK_LAYOUT_IDS",
    "SIDEWALK_ASSET_KEY",
    "RosebankRoadActor",
    "plan_rosebank_road_actors",
]
