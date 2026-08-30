"""UE adapter for deterministic visual dressing of authored districts.

Geometry, candidate selection, clearances, and actor ordering live in the pure
``district_dressing`` planner.  This module only applies those records through
the UnrealCV communicator and retains the renderer's historical compatibility
helpers.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.district_dressing import (
    DistrictActorRecord,
    _ANCHOR_CLEARANCE_CM,
    _BRIDGE_CLEARANCE_CM,
    _BUILDING_CLEARANCE_CM,
    _DISTRICT_PROP_ASSETS,
    _DISTRICT_TREE_ASSETS,
    _PROP_SCALES,
    _PROP_SPACING_CM,
    _ROUTE_CLEARANCE_CM,
    _SHELL_BUILDINGS,
    _SHELL_SPACING_CM,
    _TREE_SCALES,
    _TREE_SPACING_CM,
    _WALK_NODE_CLEARANCE_CM,
    _distance_sq,
    _minimum_segment_distance_sq,
    _offset,
    _point_to_segment_distance_sq,
    bridge_gap_polylines,
    building_actor_name as _building_actor_name,
    clear,
    district_prop_actor_name as _district_prop_actor_name,
    district_tree_actor_name as _district_tree_actor_name,
    frontages_by_block,
    inside_block,
    plan_district_actors,
    plan_prop_records,
    plan_shell_records,
    prop_candidates,
    protected_anchors,
    route_polylines,
    shell_positions,
    shell_yaw,
)
from benchmark.venue_meetup.layout import Block, Frontage

if TYPE_CHECKING:
    from benchmark.venue_meetup.scenario import Scenario
    from simworld.communicator.communicator import Communicator


# Private aliases retained for callers that imported the old module-level
# geometry helpers.  The implementations themselves are pure and now live in
# district_dressing.py.
_clear = clear
_inside_block = inside_block


class DistrictSceneRenderer:
    """Spawn static building shells plus sparse, inert district cues."""

    def __init__(self, communicator: Communicator, scenario: Scenario) -> None:
        self.communicator = communicator
        self.scenario = scenario
        self.layout = scenario.layout

    def spawn(self) -> None:
        """Apply the pure planner's records in their deterministic order."""

        for record in plan_district_actors(self.scenario):
            self._spawn_record(record)

    def _spawn_record(self, record: DistrictActorRecord) -> None:
        self._spawn_decor(
            record.actor_name,
            record.asset_key,
            record.position,
            record.yaw_deg,
            record.scale,
        )

    def _spawn_block_shells(self) -> tuple[tuple[float, float], ...]:
        """Compatibility adapter for spawning only shell records."""

        records = plan_shell_records(self.scenario)
        for record in records:
            self._spawn_record(record)
        return tuple((record.position[0], record.position[1]) for record in records)

    def _spawn_district_props(self, shell_positions: tuple[tuple[float, float], ...]) -> None:
        """Compatibility adapter for spawning tree and prop records."""

        for record in plan_prop_records(self.scenario, shell_positions):
            self._spawn_record(record)

    def _shell_positions(
        self,
        block: Block,
        venue_positions: list[tuple[float, float]],
        walk_node_positions: tuple[tuple[float, float], ...],
        *,
        frontages: tuple[Frontage, ...] = (),
        protected_anchors: tuple[tuple[tuple[float, float], float], ...] = (),
        route_polylines: tuple[tuple[tuple[float, float], ...], ...] = (),
        bridge_polylines: tuple[tuple[tuple[float, float], ...], ...] = (),
        occupied_positions: tuple[tuple[float, float], ...] = (),
    ) -> tuple[tuple[float, float], ...]:
        return shell_positions(
            block,
            venue_positions,
            walk_node_positions,
            frontages=frontages,
            protected_anchors=protected_anchors,
            route_polylines=route_polylines,
            bridge_polylines=bridge_polylines,
            occupied_positions=occupied_positions,
        )

    @staticmethod
    def _shell_yaw(
        block: Block,
        position: tuple[float, float],
        *,
        frontages: tuple[Frontage, ...] = (),
    ) -> float:
        return shell_yaw(block, position, frontages=frontages)

    def _prop_candidates(
        self,
        block: Block,
        frontages: tuple[Frontage, ...],
        occupied: tuple[tuple[float, float], ...],
        nodes: tuple[tuple[float, float], ...],
        anchors: tuple[tuple[tuple[float, float], float], ...],
        routes: tuple[tuple[tuple[float, float], ...], ...],
        limit: int,
        *,
        spacing: float = _PROP_SPACING_CM,
        bridge_polylines: tuple[tuple[tuple[float, float], ...], ...] = (),
    ) -> tuple[tuple[tuple[float, float], float], ...]:
        return prop_candidates(
            block,
            frontages,
            occupied,
            nodes,
            anchors,
            routes,
            limit,
            spacing=spacing,
            bridge_polylines=bridge_polylines,
        )

    @staticmethod
    def _offset(
        point: tuple[float, float],
        center: tuple[float, float],
        distance: float,
        lateral: float = 0.0,
    ) -> tuple[float, float]:
        return _offset(point, center, distance, lateral)

    def _frontages_by_block(self) -> dict[str, tuple[Frontage, ...]]:
        assert self.layout is not None
        return frontages_by_block(self.layout)

    def _route_polylines(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        assert self.layout is not None
        return route_polylines(self.layout)

    def _bridge_gap_polylines(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        assert self.layout is not None
        return bridge_gap_polylines(self.layout)

    def _protected_anchors(self) -> tuple[tuple[tuple[float, float], float], ...]:
        return protected_anchors(self.scenario)

    @staticmethod
    def _clear(
        point: tuple[float, float],
        anchors: tuple[tuple[tuple[float, float], float], ...],
        nodes: tuple[tuple[float, float], ...],
        routes: tuple[tuple[tuple[float, float], ...], ...],
        occupied: tuple[tuple[float, float], ...] = (),
        spacing: float = 0.0,
        *,
        bridge_polylines: tuple[tuple[tuple[float, float], ...], ...] = (),
    ) -> bool:
        return clear(
            point,
            anchors,
            nodes,
            routes,
            occupied,
            spacing,
            bridge_polylines=bridge_polylines,
        )

    @staticmethod
    def _inside_block(block: Block, point: tuple[float, float]) -> bool:
        return inside_block(block, point)

    def _spawn_decor(
        self,
        actor_name: str,
        asset_key: str,
        position: tuple[float, float, float],
        yaw_deg: float,
        scale: tuple[float, float, float],
    ) -> None:
        unrealcv = self.communicator.unrealcv
        unrealcv.spawn_bp_asset(asset_path(asset_key), actor_name)
        unrealcv.set_location(position, actor_name)
        unrealcv.set_orientation((0.0, yaw_deg, 0.0), actor_name)
        unrealcv.set_scale(scale, actor_name)
        unrealcv.set_collision(actor_name, False)
        unrealcv.set_movable(actor_name, False)

    @staticmethod
    def building_actor_name(block_id: str, shell_index: int) -> str:
        return _building_actor_name(block_id, shell_index)

    @staticmethod
    def district_prop_actor_name(block_id: str, prop_index: int) -> str:
        return _district_prop_actor_name(block_id, prop_index)

    @staticmethod
    def district_tree_actor_name(block_id: str, tree_index: int) -> str:
        return _district_tree_actor_name(block_id, tree_index)


__all__ = [
    "DistrictSceneRenderer",
    "DistrictActorRecord",
    "_DISTRICT_PROP_ASSETS",
    "_DISTRICT_TREE_ASSETS",
    "_SHELL_BUILDINGS",
    "_PROP_SCALES",
    "_TREE_SCALES",
    "_BUILDING_CLEARANCE_CM",
    "_WALK_NODE_CLEARANCE_CM",
    "_ROUTE_CLEARANCE_CM",
    "_BRIDGE_CLEARANCE_CM",
    "_ANCHOR_CLEARANCE_CM",
    "_PROP_SPACING_CM",
    "_TREE_SPACING_CM",
    "_SHELL_SPACING_CM",
    "_distance_sq",
    "_minimum_segment_distance_sq",
    "_point_to_segment_distance_sq",
    "_offset",
    "_clear",
    "_inside_block",
]
