"""Namespaced walk graph for the three-block alley playtest district."""

from __future__ import annotations

import math
from dataclasses import replace

from benchmark.venue_meetup.city_block_layout import build_city_block_layout
from benchmark.venue_meetup.connected_blocks import ConnectedBlock, ConnectedBlocksPlan
from benchmark.venue_meetup.layout import DistrictLayout, WalkEdge


def block_node_id(block_id: str, local_node_id: str) -> str:
    """Return one globally unique walk-node id within a connected district."""

    return f"{block_id}_{local_node_id}"


def _namespace_block_layout(
    block: ConnectedBlock,
) -> DistrictLayout:
    local = build_city_block_layout(
        block.plan,
        layout_id=f"{block.block_id}_city_block",
    )
    prefix = f"{block.block_id}_"
    omitted_streets = {
        "west": {"east_avenue"},
        "central": {"west_avenue", "east_avenue"},
        "east": {"west_avenue"},
    }[block.block_id]
    frontage_id = {
        frontage.frontage_id: f"{prefix}{frontage.frontage_id}"
        for frontage in local.frontages
    }
    block_id = {
        item.block_id: f"{block.block_id}_courtyard_block"
        for item in local.blocks
    }
    node_id = {
        node.node_id: block_node_id(block.block_id, node.node_id)
        for node in local.walk_nodes
    }
    return DistrictLayout(
        layout_id=f"{block.block_id}_city_block",
        schema_version=local.schema_version,
        streets=tuple(
            replace(street, street_id=f"{prefix}{street.street_id}")
            for street in local.streets
            if street.street_id not in omitted_streets
        ),
        intersections=tuple(
            replace(
                intersection,
                intersection_id=f"{prefix}{intersection.intersection_id}",
            )
            for intersection in local.intersections
        ),
        blocks=tuple(
            replace(
                item,
                block_id=block_id[item.block_id],
                frontage_ids=tuple(frontage_id[value] for value in item.frontage_ids),
            )
            for item in local.blocks
        ),
        frontages=tuple(
            replace(
                frontage,
                frontage_id=frontage_id[frontage.frontage_id],
                block_id=block_id[frontage.block_id],
                approach_node_id=(
                    node_id[frontage.approach_node_id]
                    if frontage.approach_node_id is not None
                    else None
                ),
            )
            for frontage in local.frontages
        ),
        walk_nodes=tuple(
            replace(node, node_id=node_id[node.node_id])
            for node in local.walk_nodes
        ),
        walk_edges=tuple(
            replace(
                edge,
                start_node_id=node_id[edge.start_node_id],
                end_node_id=node_id[edge.end_node_id],
            )
            for edge in local.walk_edges
        ),
    )


def build_connected_blocks_layout(
    plan: ConnectedBlocksPlan,
    *,
    layout_id: str,
) -> DistrictLayout:
    """Merge all block graphs and join aligned portal nodes with alley edges."""

    layouts = tuple(_namespace_block_layout(block) for block in plan.blocks)
    alley_edges: list[WalkEdge] = []
    for alley in plan.alleys:
        start_id = block_node_id(
            alley.first_block_id,
            f"portal_{alley.first_portal_side}_outer",
        )
        end_id = block_node_id(
            alley.second_block_id,
            f"portal_{alley.second_portal_side}_outer",
        )
        midpoint = (
            (alley.start[0] + alley.end[0]) / 2.0,
            (alley.start[1] + alley.end[1]) / 2.0,
        )
        alley_edges.append(
            WalkEdge(
                start_node_id=start_id,
                end_node_id=end_id,
                length_cm=math.dist(alley.start, midpoint)
                + math.dist(midpoint, alley.end),
                route_kind="alley",
                waypoints=(midpoint,),
            )
        )
    return DistrictLayout(
        layout_id=layout_id,
        schema_version=2,
        streets=tuple(street for layout in layouts for street in layout.streets),
        intersections=tuple(
            item for layout in layouts for item in layout.intersections
        ),
        blocks=tuple(item for layout in layouts for item in layout.blocks),
        frontages=tuple(
            frontage for layout in layouts for frontage in layout.frontages
        ),
        walk_nodes=tuple(node for layout in layouts for node in layout.walk_nodes),
        walk_edges=(
            tuple(edge for layout in layouts for edge in layout.walk_edges)
            + tuple(alley_edges)
        ),
    )


__all__ = [
    "block_node_id",
    "build_connected_blocks_layout",
]
