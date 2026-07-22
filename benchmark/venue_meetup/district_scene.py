"""Deterministic visual dressing for authored Venue Meetup districts.

DistrictLayout remains the source of truth for routes and block geometry. This
module renders the block interiors with packaged building assets while keeping
the public walk graph free of generated actors.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.layout import Block

if TYPE_CHECKING:
    from benchmark.venue_meetup.scenario import Scenario
    from simworld.communicator.communicator import Communicator


# Every choice below is used by an authored venue or landmark template, so it
# is present in the lightweight packaged UE build (unlike catalogue-only roads).
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
_BUILDING_CLEARANCE_CM = 3_400.0
_WALK_NODE_CLEARANCE_CM = 1_500.0


def _distance_sq(a: tuple[float, float], b: tuple[float, float]) -> float:
    return (a[0] - b[0]) ** 2 + (a[1] - b[1]) ** 2


class DistrictSceneRenderer:
    """Spawn dense, non-interactive building shells inside authored blocks.

    Decorative actors use the ``GEN_BP_DISTRICT_`` prefix, so environment reset
    removes them. They are intentionally kept inside block interiors: routes,
    entrances, and obstacle semantics continue to come from the existing layout
    and venue actors.
    """

    def __init__(self, communicator: Communicator, scenario: Scenario) -> None:
        self.communicator = communicator
        self.scenario = scenario
        self.layout = scenario.layout

    def spawn(self) -> None:
        """Render additional facades for every block when a layout is present."""

        if self.layout is None:
            return
        self._spawn_block_shells()

    def _spawn_block_shells(self) -> None:
        assert self.layout is not None
        walk_node_positions = tuple(node.position for node in self.layout.walk_nodes)
        venue_positions = [(venue.position[0], venue.position[1]) for venue in self.scenario.venues]

        for block_index, block in enumerate(self.layout.blocks):
            for shell_index, position in enumerate(self._shell_positions(block, venue_positions, walk_node_positions)):
                asset_key = _SHELL_BUILDINGS[(block_index * 5 + shell_index) % len(_SHELL_BUILDINGS)]
                scale_value = (0.24, 0.28, 0.32)[(block_index + shell_index) % 3]
                self._spawn_decor(
                    self.building_actor_name(block.block_id, shell_index),
                    asset_key,
                    (position[0], position[1], 0.0),
                    self._shell_yaw(block, position),
                    (scale_value, scale_value, scale_value),
                )

    def _shell_positions(
        self,
        block: Block,
        venue_positions: list[tuple[float, float]],
        walk_node_positions: tuple[tuple[float, float], ...],
    ) -> tuple[tuple[float, float], ...]:
        """Return a street-facing facade and infill parcel set for one block."""

        xs = [point[0] for point in block.footprint]
        ys = [point[1] for point in block.footprint]
        min_x, max_x = min(xs), max(xs)
        min_y, max_y = min(ys), max(ys)
        width = max_x - min_x
        height = max_y - min_y
        facade_inset = min(2_200.0, max(1_300.0, min(width, height) * 0.15))
        facade_columns = max(2, min(5, math.ceil((width - 2.0 * facade_inset) / 4_800.0)))
        facade_rows = max(2, min(4, math.ceil((height - 2.0 * facade_inset) / 4_800.0)))

        candidates: list[tuple[float, float]] = []
        # Form continuous block faces first, then add a central infill parcel.
        for column in range(facade_columns):
            x = min_x + facade_inset + (width - 2.0 * facade_inset) * (column + 0.5) / facade_columns
            candidates.extend(((x, min_y + facade_inset), (x, max_y - facade_inset)))
        for row in range(facade_rows):
            y = min_y + facade_inset + (height - 2.0 * facade_inset) * (row + 0.5) / facade_rows
            candidates.extend(((min_x + facade_inset, y), (max_x - facade_inset, y)))
        candidates.append(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))

        positions: list[tuple[float, float]] = []
        for point in candidates:
            if point in positions:
                continue
            if any(_distance_sq(point, venue) < _BUILDING_CLEARANCE_CM**2 for venue in venue_positions):
                continue
            if any(_distance_sq(point, node) < _WALK_NODE_CLEARANCE_CM**2 for node in walk_node_positions):
                continue
            positions.append(point)
            # Eight shells plus venue facades make a dense block without making
            # live scene reset overly expensive.
            if len(positions) >= 8:
                break
        if not positions:
            positions.append(((min_x + max_x) / 2.0, (min_y + max_y) / 2.0))
        return tuple(positions)

    @staticmethod
    def _shell_yaw(block: Block, position: tuple[float, float]) -> float:
        xs = [point[0] for point in block.footprint]
        ys = [point[1] for point in block.footprint]
        _, yaw_deg = min(
            (
                (abs(position[0] - min(xs)), 180.0),
                (abs(position[0] - max(xs)), 0.0),
                (abs(position[1] - min(ys)), -90.0),
                (abs(position[1] - max(ys)), 90.0),
            ),
            key=lambda item: item[0],
        )
        return yaw_deg

    def _spawn_decor(
        self,
        actor_name: str,
        asset_key: str,
        position: tuple[float, float, float],
        yaw_deg: float,
        scale: tuple[float, float, float],
    ) -> None:
        """Spawn a static visual actor without the normal collision-on default."""

        # Communicator.spawn_object deliberately enables collision for venue and
        # agent actors. Shells instead use the raw UE path and are kept static.
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
