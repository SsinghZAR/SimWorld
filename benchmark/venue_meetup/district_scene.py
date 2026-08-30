"""Deterministic visual dressing for authored Venue Meetup districts.

The layout remains the source of truth for routes, blocks, frontages, and
meeting regions.  This module only adds collision-free visual actors inside
those authored blocks.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import TYPE_CHECKING

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.layout import Block, Frontage

if TYPE_CHECKING:
    from benchmark.venue_meetup.scenario import Scenario
    from simworld.communicator.communicator import Communicator


_SHELL_BUILDINGS = (
    "BP_Building_05_C",
    "BP_Building_06_C",
    "BP_Building_20_C",
    "BP_Building_24_C",
    "BP_Building_25_C",
    "BP_Building_44_C",
    "BP_Building_87_C",
    "BP_Building_95_C",
    "BP_Building_99_C",
    "BP_Building_101_C",
    "BP_Building_123_C",
)

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
_BUILDING_CLEARANCE_CM = 3_400.0
_WALK_NODE_CLEARANCE_CM = 1_500.0
_ROUTE_CLEARANCE_CM = 1_600.0
_BRIDGE_CLEARANCE_CM = 2_200.0
_ANCHOR_CLEARANCE_CM = 1_200.0
_PROP_SPACING_CM = 1_800.0
_TREE_SPACING_CM = 2_800.0
_SHELL_SPACING_CM = 3_800.0


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


def _point_to_segment_distance_sq(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return _distance_sq(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = min(1.0, max(0.0, t))
    return _distance_sq(point, (start[0] + t * dx, start[1] + t * dy))


def _minimum_segment_distance_sq(
    point: tuple[float, float],
    polylines: Sequence[tuple[tuple[float, float], ...]],
) -> float:
    """Return the squared distance to authored route segments, not just nodes."""

    return min(
        (
            _point_to_segment_distance_sq(point, start, end)
            for polyline in polylines
            for start, end in zip(polyline, polyline[1:])
        ),
        default=math.inf,
    )


class DistrictSceneRenderer:
    """Spawn static building shells plus sparse, inert district cues."""

    def __init__(self, communicator: Communicator, scenario: Scenario) -> None:
        self.communicator = communicator
        self.scenario = scenario
        self.layout = scenario.layout

    def spawn(self) -> None:
        if self.layout is None:
            return
        shells = self._spawn_block_shells()
        self._spawn_district_props(shells)

    def _spawn_block_shells(self) -> tuple[tuple[float, float], ...]:
        assert self.layout is not None
        nodes = tuple(node.position for node in self.layout.walk_nodes)
        venues = [(venue.position[0], venue.position[1]) for venue in self.scenario.venues]
        anchors = self._protected_anchors()
        routes = self._route_polylines()
        bridge_routes = self._bridge_gap_polylines()
        by_block = self._frontages_by_block()
        spawned: list[tuple[float, float]] = []
        for block_index, block in enumerate(self.layout.blocks):
            frontages = by_block.get(block.block_id, ())
            positions = self._shell_positions(
                block,
                venues,
                nodes,
                frontages=frontages,
                protected_anchors=anchors,
                route_polylines=routes,
                bridge_polylines=bridge_routes,
                occupied_positions=spawned,
            )
            for shell_index, point in enumerate(positions):
                asset = _SHELL_BUILDINGS[(block_index * 5 + shell_index) % len(_SHELL_BUILDINGS)]
                scale = (0.24, 0.28, 0.32)[(block_index + shell_index) % 3]
                self._spawn_decor(
                    self.building_actor_name(block.block_id, shell_index),
                    asset,
                    (point[0], point[1], 0.0),
                    self._shell_yaw(block, point, frontages=frontages),
                    (scale, scale, scale),
                )
                spawned.append(point)
        return tuple(spawned)

    def _shell_positions(
        self,
        block: Block,
        venue_positions: list[tuple[float, float]],
        walk_node_positions: tuple[tuple[float, float], ...],
        *,
        frontages: Sequence[Frontage] = (),
        protected_anchors: Sequence[tuple[tuple[float, float], float]] = (),
        route_polylines: Sequence[tuple[tuple[float, float], ...]] = (),
        bridge_polylines: Sequence[tuple[tuple[float, float], ...]] = (),
        occupied_positions: Sequence[tuple[float, float]] = (),
    ) -> tuple[tuple[float, float], ...]:
        """Choose deterministic shells; an empty block is an intentional safe fallback."""

        xs, ys = zip(*block.footprint)
        min_x, max_x, min_y, max_y = min(xs), max(xs), min(ys), max(ys)
        width, height = max_x - min_x, max_y - min_y
        center = ((min_x + max_x) / 2.0, (min_y + max_y) / 2.0)
        inset = min(2_200.0, max(1_300.0, min(width, height) * 0.15))
        columns = max(2, min(5, math.ceil((width - 2.0 * inset) / 4_800.0)))
        rows = max(2, min(4, math.ceil((height - 2.0 * inset) / 4_800.0)))
        candidates: list[tuple[float, float]] = []
        for frontage in frontages:
            point = self._offset(frontage.position[:2], center, max(7_000.0, min(width, height) * 0.45))
            if min_x <= point[0] <= max_x and min_y <= point[1] <= max_y:
                candidates.append(point)
        for column in range(columns):
            x = min_x + inset + (width - 2.0 * inset) * (column + 0.5) / columns
            candidates.extend(((x, min_y + inset), (x, max_y - inset)))
        for row in range(rows):
            y = min_y + inset + (height - 2.0 * inset) * (row + 0.5) / rows
            candidates.extend(((min_x + inset, y), (max_x - inset, y)))
        candidates.append(center)

        selected: list[tuple[float, float]] = []
        for point in candidates:
            if not self._inside_block(block, point):
                continue
            if any(
                _distance_sq(point, previous) < _SHELL_SPACING_CM**2
                for previous in (*occupied_positions, *selected)
            ):
                continue
            if not self._clear(
                point,
                protected_anchors,
                walk_node_positions,
                route_polylines,
                bridge_polylines=bridge_polylines,
            ):
                continue
            if any(_distance_sq(point, venue) < _BUILDING_CLEARANCE_CM**2 for venue in venue_positions):
                continue
            selected.append(point)
            if len(selected) == 8:
                break
        return tuple(selected)

    @staticmethod
    def _shell_yaw(
        block: Block,
        position: tuple[float, float],
        *,
        frontages: Sequence[Frontage] = (),
    ) -> float:
        if frontages:
            nearest = min(frontages, key=lambda frontage: _distance_sq(position, frontage.position[:2]))
            if _distance_sq(position, nearest.position[:2]) <= 8_000.0**2:
                return float(nearest.yaw_deg)
        xs, ys = zip(*block.footprint)
        _, yaw = min(
            (
                (abs(position[0] - min(xs)), 180.0),
                (abs(position[0] - max(xs)), 0.0),
                (abs(position[1] - min(ys)), -90.0),
                (abs(position[1] - max(ys)), 90.0),
            ),
            key=lambda item: item[0],
        )
        return yaw

    def _spawn_district_props(self, shell_positions: Sequence[tuple[float, float]]) -> None:
        assert self.layout is not None
        layout_id = self.layout.layout_id.lower()
        large = "large" in layout_id or len(self.layout.blocks) >= 6
        medium = "medium" in layout_id or len(self.layout.blocks) >= 4
        limit = 4 if large else 2 if medium else 1
        assets = _DISTRICT_PROP_ASSETS if large else (
            "BP_Table_C", "BP_Hydrant_C", "BP_Trash_bin_a_C", "RoadCone_C"
        ) if medium else ("BP_Table_C", "BP_Hydrant_C")
        nodes = tuple(node.position for node in self.layout.walk_nodes)
        anchors, routes = self._protected_anchors(), self._route_polylines()
        bridge_routes = self._bridge_gap_polylines()
        by_block, occupied, serial = self._frontages_by_block(), list(shell_positions), 0
        # One scaled tree per authored block gives the camera a readable depth
        # cue without changing the route graph or introducing dynamic actors.
        for block_index, block in enumerate(self.layout.blocks):
            trees = self._prop_candidates(
                block,
                by_block.get(block.block_id, ()),
                occupied,
                nodes,
                anchors,
                routes,
                1,
                spacing=_TREE_SPACING_CM,
                bridge_polylines=bridge_routes,
            )
            if not trees:
                continue
            point, yaw = trees[0]
            asset = _DISTRICT_TREE_ASSETS[block_index % len(_DISTRICT_TREE_ASSETS)]
            self._spawn_decor(
                self.district_tree_actor_name(block.block_id, 0),
                asset,
                (point[0], point[1], 0.0),
                yaw,
                _TREE_SCALES[asset],
            )
            occupied.append(point)
        for block in self.layout.blocks:
            candidates = self._prop_candidates(
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
                self._spawn_decor(
                    self.district_prop_actor_name(block.block_id, local_index),
                    asset,
                    (point[0], point[1], 0.0),
                    yaw,
                    _PROP_SCALES[asset],
                )
                occupied.append(point)
                serial += 1

    def _prop_candidates(
        self,
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
                point = self._offset(frontage.position[:2], center, distance, lateral)
                candidates.append((point, float(frontage.yaw_deg)))
        for fy in (0.30, 0.50, 0.70):
            for fx in (0.28, 0.50, 0.72):
                point = (min_x + (max_x - min_x) * fx, min_y + (max_y - min_y) * fy)
                candidates.append((point, self._shell_yaw(block, point, frontages=frontages)))
        selected, placed = [], list(occupied)
        for point, yaw in candidates:
            if not self._inside_block(block, point):
                continue
            if not self._clear(
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

    @staticmethod
    def _offset(
        point: tuple[float, float],
        center: tuple[float, float],
        distance: float,
        lateral: float = 0.0,
    ) -> tuple[float, float]:
        dx, dy = center[0] - point[0], center[1] - point[1]
        length = math.hypot(dx, dy)
        if length <= 1e-9:
            return point
        ux, uy = dx / length, dy / length
        return (point[0] + ux * distance - uy * lateral, point[1] + uy * distance + ux * lateral)

    def _frontages_by_block(self) -> dict[str, tuple[Frontage, ...]]:
        assert self.layout is not None
        return {
            block.block_id: tuple(frontage for frontage in self.layout.frontages if frontage.block_id == block.block_id)
            for block in self.layout.blocks
        }

    def _route_polylines(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        assert self.layout is not None
        polylines = []
        for edge in self.layout.walk_edges:
            try:
                polyline = self.layout.edge_polyline(edge)
            except (KeyError, ValueError):
                continue
            if len(polyline) >= 2:
                polylines.append(polyline)
        return tuple(polylines)

    def _bridge_gap_polylines(self) -> tuple[tuple[tuple[float, float], ...], ...]:
        """Return bridge spans separately for their stronger dressing exclusion."""

        assert self.layout is not None
        polylines = []
        for edge in self.layout.walk_edges:
            if edge.route_kind != "bridge":
                continue
            try:
                polyline = self.layout.edge_polyline(edge)
            except (KeyError, ValueError):
                continue
            if len(polyline) >= 2:
                polylines.append(polyline)
        return tuple(polylines)

    def _protected_anchors(self) -> tuple[tuple[tuple[float, float], float], ...]:
        anchors = []
        for venue in self.scenario.venues:
            anchors.extend((
                (venue.region.center, float(venue.region.radius) + _ANCHOR_CLEARANCE_CM),
                ((venue.position[0], venue.position[1]), _BUILDING_CLEARANCE_CM),
            ))
            anchors.extend(
                ((entrance.position[0], entrance.position[1]), _ANCHOR_CLEARANCE_CM)
                for entrance in venue.entrances
            )
        anchors.extend(
            ((landmark.position[0], landmark.position[1]), _BUILDING_CLEARANCE_CM)
            for landmark in self.scenario.landmarks
        )
        return tuple(anchors)

    @staticmethod
    def _clear(
        point: tuple[float, float],
        anchors: Sequence[tuple[tuple[float, float], float]],
        nodes: Sequence[tuple[float, float]],
        routes: Sequence[tuple[tuple[float, float], ...]],
        occupied: Sequence[tuple[float, float]] = (),
        spacing: float = 0.0,
        *,
        bridge_polylines: Sequence[tuple[tuple[float, float], ...]] = (),
    ) -> bool:
        if any(_distance_sq(point, anchor) < clearance**2 for anchor, clearance in anchors):
            return False
        if any(_distance_sq(point, node) < _WALK_NODE_CLEARANCE_CM**2 for node in nodes):
            return False
        if _minimum_segment_distance_sq(point, routes) < _ROUTE_CLEARANCE_CM**2:
            return False
        if _minimum_segment_distance_sq(point, bridge_polylines) < _BRIDGE_CLEARANCE_CM**2:
            return False
        return not spacing or all(_distance_sq(point, other) >= spacing**2 for other in occupied)

    @staticmethod
    def _inside_block(block: Block, point: tuple[float, float]) -> bool:
        """Return whether a dressing point is contained by the authored footprint."""

        polygon = block.footprint
        if len(polygon) < 3:
            return False
        inside = False
        for index, (x1, y1) in enumerate(polygon):
            x2, y2 = polygon[(index + 1) % len(polygon)]
            if _point_to_segment_distance_sq(point, (x1, y1), (x2, y2)) <= 1e-6:
                return True
            if (y1 > point[1]) != (y2 > point[1]):
                crossing_x = (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1
                if point[0] < crossing_x:
                    inside = not inside
        return inside

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
        return f"GEN_BP_DISTRICT_BUILDING_{block_id}_{shell_index:02d}"

    @staticmethod
    def district_prop_actor_name(block_id: str, prop_index: int) -> str:
        return f"GEN_BP_DISTRICT_PROP_{block_id}_{prop_index:02d}"

    @staticmethod
    def district_tree_actor_name(block_id: str, tree_index: int) -> str:
        return f"GEN_BP_DISTRICT_TREE_{block_id}_{tree_index:02d}"
