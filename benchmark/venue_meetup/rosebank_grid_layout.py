"""Street, frontage, and walk graph for scalable Rosebank-inspired grids."""

from __future__ import annotations

import math
from dataclasses import dataclass

from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.layout import (Block, DistrictLayout, Frontage,
                                           Intersection, MeetingRegion,
                                           StreetSegment, WalkEdge, WalkNode,
                                           WalkRouteKind)
from benchmark.venue_meetup.rosebank_grid import (BLOCK_SIDE_CM,
                                                  HIGH_STREET_WIDTH_CM,
                                                  MINOR_STREET_WIDTH_CM,
                                                  OXFORD_ROAD_WIDTH_CM,
                                                  SECONDARY_STREET_WIDTH_CM,
                                                  SIDEWALK_WIDTH_CM,
                                                  RosebankGridPlan,
                                                  RosebankVenueSite,
                                                  frontage_tangent)

Point2D = tuple[float, float]
MEETING_OFFSET_CM = 500.0
MEETING_RADIUS_CM = 650.0
APPROACH_OFFSET_CM = 750.0
VENUE_SETBACK_CM = 150.0

_SIDE_OUTWARD = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}
_SIDE_YAW = {
    "north": 90.0,
    "east": 0.0,
    "south": -90.0,
    "west": 180.0,
}


@dataclass(frozen=True, slots=True)
class RosebankVenueGeometry:
    """World-space facade geometry derived from one venue site."""

    position: Point2D
    entrance: Point2D
    meeting: Point2D
    approach: Point2D
    approach_node_id: str
    street_node_id: str
    yaw_deg: float


def intersection_node_id(x_index: int, y_index: int) -> str:
    return f"ix_x{x_index}_y{y_index}"


def vertical_mid_node_id(street_index: int, block_row: int) -> str:
    return f"v{street_index}_r{block_row}_mid"


def horizontal_mid_node_id(block_column: int, street_index: int) -> str:
    return f"h{street_index}_c{block_column}_mid"


def alley_center_node_id(block_id: str) -> str:
    return f"alley_{block_id.lower()}_center"


def _street_width(
    index: int,
    *,
    vertical: bool,
    plan: RosebankGridPlan,
) -> float:
    if vertical and index == plan.primary_street_index:
        return OXFORD_ROAD_WIDTH_CM
    if not vertical and index == plan.primary_street_index:
        return HIGH_STREET_WIDTH_CM
    secondary_indices = {
        candidate
        for candidate in (
            1,
            plan.primary_street_index - 1,
            plan.primary_street_index + 1,
            plan.grid_size - 1,
        )
        if 0 < candidate < plan.grid_size
        and candidate != plan.primary_street_index
    }
    if index in secondary_indices:
        return SECONDARY_STREET_WIDTH_CM
    return MINOR_STREET_WIDTH_CM


def _edge(
    nodes: dict[str, WalkNode],
    start_id: str,
    end_id: str,
    *,
    route_kind: WalkRouteKind = "sidewalk",
) -> WalkEdge:
    start = nodes[start_id].position
    end = nodes[end_id].position
    midpoint = (
        (start[0] + end[0]) / 2.0,
        (start[1] + end[1]) / 2.0,
    )
    return WalkEdge(
        start_node_id=start_id,
        end_node_id=end_id,
        length_cm=math.dist(start, midpoint) + math.dist(midpoint, end),
        route_kind=route_kind,
        waypoints=(midpoint,),
    )


def _oriented_half_extents(site: RosebankVenueSite) -> Point2D:
    raw_x, raw_y, _raw_z = building_bbox(site.asset_key)
    half_x = raw_x * site.scale[0] / 2.0
    half_y = raw_y * site.scale[1] / 2.0
    radians = math.radians(_SIDE_YAW[site.side])
    cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
    return (
        cosine * half_x + sine * half_y,
        sine * half_x + cosine * half_y,
    )


def venue_geometry(
    plan: RosebankGridPlan,
    site: RosebankVenueSite,
) -> RosebankVenueGeometry:
    """Place a venue against its block edge and attach it to a street node."""

    block = plan.block_by_id(site.block_id)
    outward = _SIDE_OUTWARD[site.side]
    tangent = frontage_tangent(site.side)
    half_extent = BLOCK_SIDE_CM / 2.0
    boundary = (
        block.center[0]
        + outward[0] * half_extent
        + tangent[0] * site.frontage_offset_cm,
        block.center[1]
        + outward[1] * half_extent
        + tangent[1] * site.frontage_offset_cm,
    )
    world_half_x, world_half_y = _oriented_half_extents(site)
    normal_half = (
        world_half_x if site.side in {"east", "west"} else world_half_y
    )
    position = (
        boundary[0] - outward[0] * (normal_half + VENUE_SETBACK_CM),
        boundary[1] - outward[1] * (normal_half + VENUE_SETBACK_CM),
    )
    entrance = (
        boundary[0] - outward[0] * 50.0,
        boundary[1] - outward[1] * 50.0,
    )
    meeting = (
        boundary[0] + outward[0] * MEETING_OFFSET_CM,
        boundary[1] + outward[1] * MEETING_OFFSET_CM,
    )
    if site.side == "west":
        street_node_id = vertical_mid_node_id(block.column, block.row)
    elif site.side == "east":
        street_node_id = vertical_mid_node_id(block.column + 1, block.row)
    elif site.side == "south":
        street_node_id = horizontal_mid_node_id(block.column, block.row)
    else:
        street_node_id = horizontal_mid_node_id(block.column, block.row + 1)
    approach_node_id = f"approach_{site.slot_id}"
    approach = (
        boundary[0] + outward[0] * APPROACH_OFFSET_CM,
        boundary[1] + outward[1] * APPROACH_OFFSET_CM,
    )
    return RosebankVenueGeometry(
        position=position,
        entrance=entrance,
        meeting=meeting,
        approach=approach,
        approach_node_id=approach_node_id,
        street_node_id=street_node_id,
        yaw_deg=_SIDE_YAW[site.side],
    )


def _build_street_nodes(plan: RosebankGridPlan) -> dict[str, WalkNode]:
    nodes: dict[str, WalkNode] = {}
    for x_index, x in enumerate(plan.street_x):
        for y_index, y in enumerate(plan.street_y):
            node_id = intersection_node_id(x_index, y_index)
            nodes[node_id] = WalkNode(node_id, (x, y), "intersection")
        for row in range(plan.grid_size):
            node_id = vertical_mid_node_id(x_index, row)
            nodes[node_id] = WalkNode(
                node_id,
                (x, plan.block_at(row, 0).center[1]),
                "sidewalk",
            )
    for y_index, y in enumerate(plan.street_y):
        for column in range(plan.grid_size):
            node_id = horizontal_mid_node_id(column, y_index)
            nodes[node_id] = WalkNode(
                node_id,
                (plan.block_at(0, column).center[0], y),
                "sidewalk",
            )
    for block in plan.blocks:
        if not block.alley_axes:
            continue
        node_id = alley_center_node_id(block.block_id)
        nodes[node_id] = WalkNode(node_id, block.center, "sidewalk")
    for site in plan.venue_sites:
        geometry = venue_geometry(plan, site)
        nodes[geometry.approach_node_id] = WalkNode(
            geometry.approach_node_id,
            geometry.approach,
            "sidewalk",
        )
    return nodes


def _street_edges(
    plan: RosebankGridPlan,
    nodes: dict[str, WalkNode],
) -> list[WalkEdge]:
    edges: list[WalkEdge] = []
    for x_index in range(plan.grid_size + 1):
        node_ids = [
            *(
                intersection_node_id(x_index, y_index)
                for y_index in range(plan.grid_size + 1)
            ),
            *(
                vertical_mid_node_id(x_index, row)
                for row in range(plan.grid_size)
            ),
        ]
        node_ids.sort(key=lambda node_id: nodes[node_id].position[1])
        edges.extend(
            _edge(nodes, start_id, end_id)
            for start_id, end_id in zip(node_ids, node_ids[1:])
        )
    for y_index in range(plan.grid_size + 1):
        node_ids = [
            *(
                intersection_node_id(x_index, y_index)
                for x_index in range(plan.grid_size + 1)
            ),
            *(
                horizontal_mid_node_id(column, y_index)
                for column in range(plan.grid_size)
            ),
        ]
        node_ids.sort(key=lambda node_id: nodes[node_id].position[0])
        edges.extend(
            _edge(nodes, start_id, end_id)
            for start_id, end_id in zip(node_ids, node_ids[1:])
        )
    return edges


def _alley_edges(
    plan: RosebankGridPlan,
    nodes: dict[str, WalkNode],
) -> list[WalkEdge]:
    edges: list[WalkEdge] = []
    for block in plan.blocks:
        center_id = alley_center_node_id(block.block_id)
        if "horizontal" in block.alley_axes:
            edges.extend(
                (
                    _edge(
                        nodes,
                        vertical_mid_node_id(block.column, block.row),
                        center_id,
                        route_kind="alley",
                    ),
                    _edge(
                        nodes,
                        center_id,
                        vertical_mid_node_id(block.column + 1, block.row),
                        route_kind="alley",
                    ),
                )
            )
        if "vertical" in block.alley_axes:
            edges.extend(
                (
                    _edge(
                        nodes,
                        horizontal_mid_node_id(block.column, block.row),
                        center_id,
                        route_kind="alley",
                    ),
                    _edge(
                        nodes,
                        center_id,
                        horizontal_mid_node_id(block.column, block.row + 1),
                        route_kind="alley",
                    ),
                )
            )
    return edges


def _frontage_edges(
    plan: RosebankGridPlan,
    nodes: dict[str, WalkNode],
) -> list[WalkEdge]:
    return [
        _edge(
            nodes,
            venue_geometry(plan, site).street_node_id,
            venue_geometry(plan, site).approach_node_id,
        )
        for site in plan.venue_sites
    ]


def build_rosebank_grid_layout(
    plan: RosebankGridPlan,
    *,
    layout_id: str,
) -> DistrictLayout:
    """Return all blocks, hierarchical streets, and alley shortcuts."""

    nodes = _build_street_nodes(plan)
    frontages: list[Frontage] = []
    frontage_ids_by_block: dict[str, list[str]] = {
        block.block_id: [] for block in plan.blocks
    }
    for site in plan.venue_sites:
        geometry = venue_geometry(plan, site)
        frontage_id = f"frontage_{site.slot_id}"
        frontage_ids_by_block[site.block_id].append(frontage_id)
        frontages.append(
            Frontage(
                frontage_id=frontage_id,
                block_id=site.block_id,
                position=(*geometry.position, 0.0),
                yaw_deg=geometry.yaw_deg,
                entrance_point=(*geometry.entrance, 0.0),
                meeting_region=MeetingRegion(
                    center=geometry.meeting,
                    radius=MEETING_RADIUS_CM,
                ),
                venue_slot_id=site.slot_id,
                approach_node_id=geometry.approach_node_id,
                access_path=(geometry.meeting,),
            )
        )

    half_extent = BLOCK_SIDE_CM / 2.0
    blocks = tuple(
        Block(
            block_id=block.block_id,
            footprint=(
                (block.center[0] - half_extent, block.center[1] - half_extent),
                (block.center[0] + half_extent, block.center[1] - half_extent),
                (block.center[0] + half_extent, block.center[1] + half_extent),
                (block.center[0] - half_extent, block.center[1] + half_extent),
            ),
            frontage_ids=tuple(frontage_ids_by_block[block.block_id]),
            visual_style=block.visual_style,
            # This large grid uses its own bounded cell-massing planner. Avoid
            # adding the generic 24-shell-per-block renderer on top of it.
            shell_target=0,
        )
        for block in plan.blocks
    )
    streets = tuple(
        [
            StreetSegment(
                street_id=name,
                start=(x, plan.street_y[0]),
                end=(x, plan.street_y[-1]),
                width_cm=_street_width(index, vertical=True, plan=plan),
                sidewalk_width_cm=SIDEWALK_WIDTH_CM,
            )
            for index, (name, x) in enumerate(
                zip(plan.vertical_street_names, plan.street_x)
            )
        ]
        + [
            StreetSegment(
                street_id=name,
                start=(plan.street_x[0], y),
                end=(plan.street_x[-1], y),
                width_cm=_street_width(index, vertical=False, plan=plan),
                sidewalk_width_cm=SIDEWALK_WIDTH_CM,
            )
            for index, (name, y) in enumerate(
                zip(plan.horizontal_street_names, plan.street_y)
            )
        ]
    )
    intersections = tuple(
        Intersection(
            intersection_id=f"landmark_gateway_{block.block_id.lower()}",
            position=(
                plan.street_x[block.column],
                plan.street_y[block.row],
            ),
            landmark_id=f"landmark_{block.landmark_role}",
        )
        for block in plan.blocks
        if block.landmark_role is not None
    )
    walk_edges = (
        *_street_edges(plan, nodes),
        *_alley_edges(plan, nodes),
        *_frontage_edges(plan, nodes),
    )
    return DistrictLayout(
        layout_id=layout_id,
        schema_version=2,
        streets=streets,
        intersections=intersections,
        blocks=blocks,
        frontages=tuple(frontages),
        walk_nodes=tuple(nodes.values()),
        walk_edges=walk_edges,
    )


__all__ = [
    "MEETING_OFFSET_CM",
    "MEETING_RADIUS_CM",
    "APPROACH_OFFSET_CM",
    "RosebankVenueGeometry",
    "alley_center_node_id",
    "build_rosebank_grid_layout",
    "horizontal_mid_node_id",
    "intersection_node_id",
    "venue_geometry",
    "vertical_mid_node_id",
]
