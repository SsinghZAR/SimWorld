"""Pure deterministic planning for authored Venue Meetup district dressing.

The planner in this module only consumes scenario/layout geometry and returns
ordered actor records.  It deliberately has no UnrealCV/communicator dependency;
``district_scene.DistrictSceneRenderer`` is the small adapter that applies these
records to the live scene.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import TYPE_CHECKING

from benchmark.venue_meetup.district_geometry import (
    DistrictShellFootprint,
    _ANCHOR_CLEARANCE_CM,
    _BRIDGE_CLEARANCE_CM,
    _BUILDING_CLEARANCE_CM,
    _ROUTE_CLEARANCE_CM,
    _SHELL_ASSET_MARGIN_CM,
    _SHELL_EDGE_END_GAP_CM,
    _SHELL_EDGE_SETBACK_CM,
    _SHELL_FRONTAGE_BUFFER_CM,
    _SHELL_FRONTAGE_GAP_CM,
    _SHELL_RHYTHM,
    _SHELL_SCALES,
    _SHELL_SEAM_GAP_CM,
    _SHELL_TARGET_LARGE,
    _SHELL_TARGET_MEDIUM,
    _WALK_NODE_CLEARANCE_CM,
    _augment_shell_shortfall,
    _distance_sq,
    _make_shell_placement,
    _minimum_segment_distance_sq,
    _offset,
    _oriented_half_extents,
    _tile_block,
    bridge_gap_polylines,
    clear,
    frontages_by_block,
    inside_block,
    protected_anchors,
    route_polylines,
    shell_positions,
    shell_yaw,
    shell_protected_bounds,
)
from benchmark.venue_meetup.layout import Block, Frontage

if TYPE_CHECKING:
    from benchmark.venue_meetup.scenario import Scenario


# Do not add catalogue-only road blueprints here: they are unavailable on the
# packaged empty map and have previously crashed Unreal when spawned.
_DISTRICT_PROP_ASSETS = (
    "RoadBlocker_C",
    "RoadCone_C",
    "BP_Table_C",
    "BP_Table2_C",
    "BP_Can_C",
    "BP_Soda1_C",
    "BP_Trash_bin_a_C",
    "BP_Hydrant_C",
)
# These two catalogue assets were live-probed on the packaged map.  They are
# readable at district scale while the scooter/cart/box candidates are not.
_DISTRICT_TREE_ASSETS = ("BP_Tree1_C", "BP_Tree2_C")
_PROP_SCALES = {
    "RoadBlocker_C": (0.70, 0.70, 0.70),
    "RoadCone_C": (0.58, 0.58, 0.58),
    "BP_Table_C": (0.78, 0.78, 0.78),
    "BP_Table2_C": (0.78, 0.78, 0.78),
    "BP_Can_C": (0.40, 0.40, 0.40),
    "BP_Soda1_C": (0.40, 0.40, 0.40),
    "BP_Trash_bin_a_C": (0.68, 0.68, 0.68),
    "BP_Hydrant_C": (0.70, 0.70, 0.70),
}
_TREE_SCALES = {
    "BP_Tree1_C": (1.45, 1.45, 1.45),
    "BP_Tree2_C": (1.35, 1.35, 1.35),
}
_PROP_SPACING_CM = 1_800.0
_TREE_SPACING_CM = 2_800.0
@dataclass(frozen=True, slots=True)
class DistrictActorRecord:
    """One inert visual actor planned for a district.

    ``position`` is the exact three-dimensional Unreal location supplied by the
    renderer.  The adapter applies the fixed inert-actor policy when spawning.
    """

    actor_name: str
    asset_key: str
    position: tuple[float, float, float]
    yaw_deg: float
    scale: tuple[float, float, float]
    collision: bool = False
    footprint: DistrictShellFootprint | None = None


def prop_candidates(
    block: Block,
    frontages: Sequence[Frontage],
    occupied: Sequence[tuple[float, float]],
    nodes: Sequence[tuple[float, float]],
    anchors: Sequence[tuple[tuple[float, float], float]],
    routes: Sequence[tuple[tuple[float, float], ...]],
    limit: int,
    *,
    spacing: float = _PROP_SPACING_CM,
    bridge_polylines: Sequence[tuple[tuple[float, float], ...]] = (),
) -> tuple[tuple[tuple[float, float], float], ...]:
    xs, ys = zip(*block.footprint)
    min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
    center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
    candidates: list[tuple[tuple[float, float], float]] = []
    for frontage in frontages:
        # Prefer cues on the street-facing side of a frontage; fall back
        # inward where a meeting region or block boundary leaves no room.
        for distance, lateral in ((-6_500.0, 0.0), (-6_500.0, 2_600.0), (2_600.0, 0.0), (2_600.0, 2_600.0)):
            point = _offset(frontage.position[:2], center, distance, lateral)
            candidates.append((point, float(frontage.yaw_deg)))
    for fy in (0.30, 0.50, 0.70):
        for fx in (0.28, 0.50, 0.72):
            point = (min_x + (max_x - min_x) * fx, min_y + (max_y - min_y) * fy)
            candidates.append((point, shell_yaw(block, point, frontages=frontages)))
    selected, placed = [], list(occupied)
    for point, yaw in candidates:
        if not inside_block(block, point):
            continue
        if not clear(
            point,
            anchors,
            nodes,
            routes,
            placed,
            spacing,
            bridge_polylines=bridge_polylines,
        ):
            continue
        selected.append((point, yaw))
        placed.append(point)
        if len(selected) == limit:
            break
    return tuple(selected)


def _tree_fallback_candidate(
    block: Block,
    *,
    occupied: Sequence[tuple[float, float]],
    nodes: Sequence[tuple[float, float]],
    anchors: Sequence[tuple[tuple[float, float], float]],
    routes: Sequence[tuple[tuple[float, float], ...]],
    bridge_polylines: Sequence[tuple[tuple[float, float], ...]],
) -> tuple[tuple[float, float], float] | None:
    """Find a deterministic sparse tree point when frontage candidates are full.

    Shells intentionally consume most perimeter slots.  A tree is non-colliding
    dressing, so when the normal frontage/interior candidates are blocked we
    scan the block interior at a coarse fixed lattice while retaining the same
    venue, node, route, and bridge clearances as every other prop.
    """

    if len(block.footprint) < 3:
        return None
    xs, ys = zip(*block.footprint)
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    step = 1_000.0
    candidates: list[tuple[float, float]] = []
    for iy, y in enumerate(range(int(math.ceil(min_y)), int(math.floor(max_y)) + 1, int(step))):
        # Alternate the x scan direction by row so the fallback remains stable
        # but does not stack every block's first tree on one corner.
        row = list(range(int(math.ceil(min_x)), int(math.floor(max_x)) + 1, int(step)))
        if iy % 2:
            row.reverse()
        candidates.extend((float(x), float(y)) for x in row)
    for point in candidates:
        if not inside_block(block, point):
            continue
        if clear(
            point,
            anchors,
            nodes,
            routes,
            occupied,
            _TREE_SPACING_CM,
            bridge_polylines=bridge_polylines,
        ):
            return point, shell_yaw(block, point)
    return None


def building_actor_name(block_id: str, shell_index: int) -> str:
    return f"GEN_BP_DISTRICT_BUILDING_{block_id}_{shell_index:02d}"


def district_prop_actor_name(block_id: str, prop_index: int) -> str:
    return f"GEN_BP_DISTRICT_PROP_{block_id}_{prop_index:02d}"


def district_tree_actor_name(block_id: str, tree_index: int) -> str:
    return f"GEN_BP_DISTRICT_TREE_{block_id}_{tree_index:02d}"


def plan_shell_records(scenario: Scenario) -> tuple[DistrictActorRecord, ...]:
    """Plan dense perimeter shell records in deterministic block/edge order."""

    layout = scenario.layout
    if layout is None:
        return ()
    nodes = tuple(node.position for node in layout.walk_nodes)
    # Shells use measured AABBs and small authored point margins.  The legacy
    # 3.4m venue-center/1.2m anchor buffers are for props and trees; retaining
    # them here would reject most valid perimeter frontage slots.
    anchors = tuple(
        [
            (venue.region.center, float(venue.region.radius) + _SHELL_FRONTAGE_BUFFER_CM),
            *((entrance.position[:2], _SHELL_ASSET_MARGIN_CM) for entrance in venue.entrances),
        ]
        for venue in scenario.venues
    )
    anchors = tuple(item for group in anchors for item in group)
    protected_bounds = shell_protected_bounds(scenario)
    routes = route_polylines(layout)
    bridge_routes = bridge_gap_polylines(layout)
    by_block = frontages_by_block(layout)
    large = "large" in layout.layout_id.lower() or len(layout.blocks) >= 6
    target_count = _SHELL_TARGET_LARGE if large else _SHELL_TARGET_MEDIUM
    venue_by_slot = {venue.slot_id: venue for venue in scenario.venues}
    spawned_placements: list[_ShellPlacement] = []
    records: list[DistrictActorRecord] = []
    for block_index, block in enumerate(layout.blocks):
        frontages = by_block.get(block.block_id, ())
        block_target = (
            int(block.shell_target)
            if block.shell_target is not None
            else target_count
        )
        if block_target < 0:
            raise ValueError(
                f"Block {block.block_id!r} shell_target must be non-negative: "
                f"{block.shell_target!r}"
            )
        if block_target == 0:
            continue
        placements = _tile_block(
            block,
            frontages=frontages,
            venue_by_slot=venue_by_slot,
            venue_positions=(),
            walk_node_positions=nodes,
            protected_anchors=anchors,
            route_polylines=routes,
            bridge_polylines=bridge_routes,
            occupied_placements=spawned_placements,
            occupied_positions=(),
            protected_bounds=protected_bounds,
            target_count=block_target,
            block_index=block_index,
        )
        placements = _augment_shell_shortfall(
            block,
            placements,
            minimum_count=block_target,
            frontages=frontages,
            venue_by_slot=venue_by_slot,
            walk_node_positions=nodes,
            protected_anchors=anchors,
            route_polylines=routes,
            bridge_polylines=bridge_routes,
            occupied_placements=spawned_placements,
            block_index=block_index,
            protected_bounds=protected_bounds,
        )
        for shell_index, placement in enumerate(placements):
            actor_name = building_actor_name(block.block_id, shell_index)
            footprint = replace(placement.footprint, actor_name=actor_name)
            records.append(
                DistrictActorRecord(
                    actor_name=actor_name,
                    asset_key=placement.asset_key,
                    position=(placement.point[0], placement.point[1], 0.0),
                    yaw_deg=placement.yaw_deg,
                    scale=placement.scale,
                    collision=True,
                    footprint=footprint,
                )
            )
            spawned_placements.append(replace(placement, footprint=footprint))
    return tuple(records)


def plan_shell_footprints(scenario: Scenario) -> tuple[DistrictShellFootprint, ...]:
    """Return the authoritative conservative footprint for every shell."""

    return tuple(
        record.footprint
        for record in plan_shell_records(scenario)
        if record.footprint is not None
    )


def plan_prop_records(
    scenario: Scenario,
    shell_positions: Sequence[tuple[float, float]],
) -> tuple[DistrictActorRecord, ...]:
    """Plan tree and prop records after the supplied shell positions."""

    layout = scenario.layout
    if layout is None:
        return ()
    layout_id = layout.layout_id.lower()
    large = "large" in layout_id or len(layout.blocks) >= 6
    medium = "medium" in layout_id or len(layout.blocks) >= 4
    # Keep street furniture proportional to district area rather than block
    # count. The canal redesign doubles the number of real blocks; retaining
    # four props and one tree per block would add visual clutter and inflate
    # live reset times without improving navigation cues.
    limit = 2 if large or medium else 1
    assets = _DISTRICT_PROP_ASSETS if large else (
        "BP_Table_C", "BP_Hydrant_C", "BP_Trash_bin_a_C", "RoadCone_C"
    ) if medium else ("BP_Table_C", "BP_Hydrant_C")
    nodes = tuple(node.position for node in layout.walk_nodes)
    anchors, routes = protected_anchors(scenario), route_polylines(layout)
    bridge_routes = bridge_gap_polylines(layout)
    # Shells are solid but props/trees are inert visual cues; do not let the
    # shell center list erase the authored furniture budget at dense counts.
    # Tree candidates still use local shell positions below to avoid obvious
    # visual stacking, while props share only the tree/prop occupancy list.
    by_block, occupied, serial = frontages_by_block(layout), [], 0
    # Keep tree placement local to its owning block.  A global shell-spacing
    # exclusion can erase a valid interior point when neighbouring blocks share
    # a street seam; the tree is non-colliding and its local spacing/route
    # checks are the relevant constraints.
    shell_positions_by_block = {
        block.block_id: [point for point in shell_positions if inside_block(block, point)]
        for block in layout.blocks
    }
    records: list[DistrictActorRecord] = []
    # The large map uses paired narrow blocks, so one tree per pair preserves
    # the original six-tree budget. The medium map keeps one per block.
    dressing_blocks = tuple(
        block for block in layout.blocks if block.shell_target != 0
    )
    tree_blocks = dressing_blocks[::2] if large else dressing_blocks
    for block_index, block in enumerate(tree_blocks):
        tree_occupied = [*shell_positions_by_block.get(block.block_id, ()), *(
            (record.position[0], record.position[1])
            for record in records
            if "DISTRICT_TREE_" in record.actor_name
        )]
        trees = prop_candidates(
            block,
            by_block.get(block.block_id, ()),
            tree_occupied,
            nodes,
            anchors,
            routes,
            1,
            spacing=_TREE_SPACING_CM,
            bridge_polylines=bridge_routes,
        )
        if not trees:
            fallback_tree = _tree_fallback_candidate(
                block,
                occupied=tree_occupied,
                nodes=nodes,
                anchors=anchors,
                routes=routes,
                bridge_polylines=bridge_routes,
            )
            trees = (fallback_tree,) if fallback_tree is not None else ()
        if not trees:
            continue
        point, yaw = trees[0]
        asset = _DISTRICT_TREE_ASSETS[block_index % len(_DISTRICT_TREE_ASSETS)]
        records.append(
            DistrictActorRecord(
                actor_name=district_tree_actor_name(block.block_id, 0),
                asset_key=asset,
                position=(point[0], point[1], 0.0),
                yaw_deg=yaw,
                scale=_TREE_SCALES[asset],
            )
        )
        occupied.append(point)
    for block in dressing_blocks:
        candidates = prop_candidates(
            block,
            by_block.get(block.block_id, ()),
            occupied,
            nodes,
            anchors,
            routes,
            limit,
            bridge_polylines=bridge_routes,
        )
        for local_index, (point, yaw) in enumerate(candidates):
            # Keep one table at every frontage so the sparse dressing reads
            # as street furniture; the remaining cues cycle by layout size.
            asset = (
                "BP_Table_C"
                if local_index == 0
                else assets[(serial + local_index - 1) % len(assets)]
            )
            records.append(
                DistrictActorRecord(
                    actor_name=district_prop_actor_name(block.block_id, local_index),
                    asset_key=asset,
                    position=(point[0], point[1], 0.0),
                    yaw_deg=yaw,
                    scale=_PROP_SCALES[asset],
                )
            )
            occupied.append(point)
            serial += 1
    return tuple(records)


def plan_district_actors(scenario: Scenario) -> tuple[DistrictActorRecord, ...]:
    """Return every inert district actor in renderer spawn order."""

    shells = plan_shell_records(scenario)
    shell_xy = tuple((record.position[0], record.position[1]) for record in shells)
    return (*shells, *plan_prop_records(scenario, shell_xy))


__all__ = [
    "DistrictActorRecord",
    "plan_district_actors",
    "plan_shell_records",
    "plan_prop_records",
    "building_actor_name",
    "district_prop_actor_name",
    "district_tree_actor_name",
    "_DISTRICT_PROP_ASSETS",
    "_DISTRICT_TREE_ASSETS",
    "_PROP_SCALES",
    "_TREE_SCALES",
    "_BUILDING_CLEARANCE_CM",
    "_WALK_NODE_CLEARANCE_CM",
    "_ROUTE_CLEARANCE_CM",
    "_BRIDGE_CLEARANCE_CM",
    "_ANCHOR_CLEARANCE_CM",
    "_PROP_SPACING_CM",
    "_TREE_SPACING_CM",
    "clear",
    "inside_block",
    "shell_positions",
    "shell_yaw",
    "prop_candidates",
    "frontages_by_block",
    "route_polylines",
    "bridge_gap_polylines",
    "protected_anchors",
]
