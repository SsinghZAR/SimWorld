"""Authored walk graph and frontage geometry for the four-entry city block."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark.venue_meetup.busy_street import BusyStreetBuilding
from benchmark.venue_meetup.city_block import (
    BlockSide,
    CityBlockPlan,
    SIDE_OUTWARD,
)
from benchmark.venue_meetup.layout import (
    Block,
    DistrictLayout,
    Frontage,
    Intersection,
    MeetingRegion,
    StreetSegment,
    WalkEdge,
    WalkNode,
    WalkRouteKind,
)

STREET_WIDTH_CM = 2_400.0
SIDEWALK_WIDTH_CM = 800.0
MEETING_OFFSET_CM = 350.0
APPROACH_OFFSET_CM = 520.0
OUTER_WALK_OFFSET_CM = 750.0
PORTAL_INNER_OFFSET_CM = 1_000.0
COURTYARD_RING_OFFSET_CM = 700.0


@dataclass(frozen=True, slots=True)
class BlockFrontageGeometry:
    """Side-aware public access points derived from one facade placement."""

    side: BlockSide
    outward: tuple[float, float]
    boundary: tuple[float, float]
    meeting: tuple[float, float]
    approach: tuple[float, float]
    sidewalk: tuple[float, float]


def offset_point(
    point: tuple[float, float],
    direction: tuple[float, float],
    distance_cm: float,
) -> tuple[float, float]:
    """Offset a 2D point along a unit direction."""

    return (
        point[0] + direction[0] * distance_cm,
        point[1] + direction[1] * distance_cm,
    )


def venue_slot_id(building: BusyStreetBuilding) -> str:
    """Return the stable scenario slot id for an interactable facade."""

    if building.venue_id is None:
        raise ValueError("Residential facade has no venue slot id")
    return building.venue_id.removeprefix("venue_")


def frontage_geometry(
    plan: CityBlockPlan,
    building: BusyStreetBuilding,
) -> BlockFrontageGeometry:
    """Derive entrance, meeting, approach, and sidewalk points once."""

    side = plan.side_for_building(building.placement.index)
    outward = SIDE_OUTWARD[side]
    placement = building.placement
    normal_offset = placement.normal_half_extent_cm + plan.setback_cm
    boundary = offset_point(placement.position, outward, normal_offset)
    return BlockFrontageGeometry(
        side=side,
        outward=outward,
        boundary=boundary,
        meeting=offset_point(boundary, outward, MEETING_OFFSET_CM),
        approach=offset_point(boundary, outward, APPROACH_OFFSET_CM),
        sidewalk=offset_point(boundary, outward, OUTER_WALK_OFFSET_CM),
    )


def _frontage_id(building: BusyStreetBuilding) -> str:
    return f"frontage_{venue_slot_id(building)}"


def _approach_id(building: BusyStreetBuilding) -> str:
    return f"approach_{venue_slot_id(building)}"


def _sidewalk_id(building: BusyStreetBuilding) -> str:
    return f"sidewalk_{venue_slot_id(building)}"


def _edge(
    nodes: dict[str, WalkNode],
    start_id: str,
    end_id: str,
    *,
    route_kind: WalkRouteKind = "sidewalk",
) -> WalkEdge:
    start = nodes[start_id].position
    end = nodes[end_id].position
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    return WalkEdge(
        start_node_id=start_id,
        end_node_id=end_id,
        length_cm=math.dist(start, midpoint) + math.dist(midpoint, end),
        route_kind=route_kind,
        waypoints=(midpoint,),
    )


def _ring_sort_key(side: BlockSide, node: WalkNode) -> float:
    if side == "north":
        return node.position[0]
    if side == "east":
        return -node.position[1]
    if side == "south":
        return -node.position[0]
    return node.position[1]


def build_city_block_layout(
    plan: CityBlockPlan,
    *,
    layout_id: str,
) -> DistrictLayout:
    """Build connected outer-sidewalk, four-portal, and courtyard rings."""

    venue_buildings = tuple(
        building for building in plan.buildings if building.venue_id is not None
    )
    geometries = {
        building.placement.index: frontage_geometry(plan, building)
        for building in venue_buildings
    }
    frontages: list[Frontage] = []
    nodes: dict[str, WalkNode] = {}
    side_ring_nodes: dict[BlockSide, list[str]] = {
        side: [] for side in ("north", "east", "south", "west")
    }

    for building in venue_buildings:
        geometry = geometries[building.placement.index]
        frontages.append(
            Frontage(
                frontage_id=_frontage_id(building),
                block_id="block_courtyard",
                position=(*building.placement.position, 0.0),
                yaw_deg=building.placement.yaw_deg,
                entrance_point=(*geometry.boundary, 0.0),
                meeting_region=MeetingRegion(
                    center=geometry.meeting,
                    radius=350.0,
                ),
                venue_slot_id=venue_slot_id(building),
                approach_node_id=_approach_id(building),
                access_path=(geometry.meeting,),
            )
        )
        nodes[_sidewalk_id(building)] = WalkNode(
            _sidewalk_id(building),
            geometry.sidewalk,
            "sidewalk",
        )
        nodes[_approach_id(building)] = WalkNode(
            _approach_id(building),
            geometry.approach,
            "sidewalk",
        )
        side_ring_nodes[geometry.side].append(_sidewalk_id(building))

    outer = plan.half_extent_cm + OUTER_WALK_OFFSET_CM
    corner_positions = {
        "outer_north_west": (-outer, outer),
        "outer_north_east": (outer, outer),
        "outer_south_east": (outer, -outer),
        "outer_south_west": (-outer, -outer),
    }
    for node_id, position in corner_positions.items():
        nodes[node_id] = WalkNode(node_id, position, "intersection")

    for portal in plan.portals:
        outer_id = f"{portal.portal_id}_outer"
        threshold_id = f"{portal.portal_id}_threshold"
        inner_id = f"{portal.portal_id}_inner"
        nodes[outer_id] = WalkNode(
            outer_id,
            portal.offset_position(OUTER_WALK_OFFSET_CM),
            "spawn" if portal.side in {"east", "west"} else "sidewalk",
        )
        nodes[threshold_id] = WalkNode(
            threshold_id,
            portal.boundary_position,
            "crossing",
        )
        nodes[inner_id] = WalkNode(
            inner_id,
            portal.offset_position(-PORTAL_INNER_OFFSET_CM),
            "sidewalk",
        )
        side_ring_nodes[portal.side].append(outer_id)

    courtyard_positions = {
        "courtyard_north": (0.0, COURTYARD_RING_OFFSET_CM),
        "courtyard_east": (COURTYARD_RING_OFFSET_CM, 0.0),
        "courtyard_south": (0.0, -COURTYARD_RING_OFFSET_CM),
        "courtyard_west": (-COURTYARD_RING_OFFSET_CM, 0.0),
    }
    for node_id, position in courtyard_positions.items():
        nodes[node_id] = WalkNode(node_id, position, "sidewalk")

    side_corners: dict[BlockSide, tuple[str, str]] = {
        "north": ("outer_north_west", "outer_north_east"),
        "east": ("outer_north_east", "outer_south_east"),
        "south": ("outer_south_east", "outer_south_west"),
        "west": ("outer_south_west", "outer_north_west"),
    }
    edges: list[WalkEdge] = []
    for side in ("north", "east", "south", "west"):
        start_corner, end_corner = side_corners[side]
        ordered = sorted(
            side_ring_nodes[side],
            key=lambda node_id: _ring_sort_key(side, nodes[node_id]),
        )
        chain = [start_corner, *ordered, end_corner]
        edges.extend(
            _edge(nodes, left, right)
            for left, right in zip(chain, chain[1:])
        )

    edges.extend(
        _edge(nodes, _sidewalk_id(building), _approach_id(building))
        for building in venue_buildings
    )
    courtyard_by_side = {
        "north": "courtyard_north",
        "east": "courtyard_east",
        "south": "courtyard_south",
        "west": "courtyard_west",
    }
    for portal in plan.portals:
        outer_id = f"{portal.portal_id}_outer"
        threshold_id = f"{portal.portal_id}_threshold"
        inner_id = f"{portal.portal_id}_inner"
        edges.extend(
            (
                _edge(nodes, outer_id, threshold_id, route_kind="alley"),
                _edge(nodes, threshold_id, inner_id, route_kind="alley"),
                _edge(
                    nodes,
                    inner_id,
                    courtyard_by_side[portal.side],
                    route_kind="alley",
                ),
            )
        )
    courtyard_cycle = (
        "courtyard_north",
        "courtyard_east",
        "courtyard_south",
        "courtyard_west",
        "courtyard_north",
    )
    edges.extend(
        _edge(nodes, left, right, route_kind="alley")
        for left, right in zip(courtyard_cycle, courtyard_cycle[1:])
    )

    street_extent = (
        plan.half_extent_cm + SIDEWALK_WIDTH_CM + STREET_WIDTH_CM / 2.0
    )
    return DistrictLayout(
        layout_id=layout_id,
        schema_version=2,
        streets=(
            StreetSegment(
                "north_avenue",
                (-street_extent, street_extent),
                (street_extent, street_extent),
                STREET_WIDTH_CM,
                SIDEWALK_WIDTH_CM,
            ),
            StreetSegment(
                "east_avenue",
                (street_extent, street_extent),
                (street_extent, -street_extent),
                STREET_WIDTH_CM,
                SIDEWALK_WIDTH_CM,
            ),
            StreetSegment(
                "south_avenue",
                (street_extent, -street_extent),
                (-street_extent, -street_extent),
                STREET_WIDTH_CM,
                SIDEWALK_WIDTH_CM,
            ),
            StreetSegment(
                "west_avenue",
                (-street_extent, -street_extent),
                (-street_extent, street_extent),
                STREET_WIDTH_CM,
                SIDEWALK_WIDTH_CM,
            ),
        ),
        intersections=tuple(
            Intersection(node_id.removeprefix("outer_"), position)
            for node_id, position in corner_positions.items()
        ),
        blocks=(
            Block(
                block_id="block_courtyard",
                footprint=(
                    (-plan.half_extent_cm, -plan.half_extent_cm),
                    (plan.half_extent_cm, -plan.half_extent_cm),
                    (plan.half_extent_cm, plan.half_extent_cm),
                    (-plan.half_extent_cm, plan.half_extent_cm),
                ),
                frontage_ids=tuple(
                    frontage.frontage_id for frontage in frontages
                ),
                visual_style="authored_courtyard_block",
                shell_target=0,
            ),
        ),
        frontages=tuple(frontages),
        walk_nodes=tuple(nodes.values()),
        walk_edges=tuple(edges),
    )


__all__ = [
    "APPROACH_OFFSET_CM",
    "BlockFrontageGeometry",
    "COURTYARD_RING_OFFSET_CM",
    "MEETING_OFFSET_CM",
    "OUTER_WALK_OFFSET_CM",
    "PORTAL_INNER_OFFSET_CM",
    "SIDEWALK_WIDTH_CM",
    "STREET_WIDTH_CM",
    "build_city_block_layout",
    "frontage_geometry",
    "offset_point",
    "venue_slot_id",
]
