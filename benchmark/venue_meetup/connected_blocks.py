"""Composition of three four-entry blocks joined by pedestrian alleys."""

from __future__ import annotations

import math
from dataclasses import dataclass, replace

from benchmark.venue_meetup.busy_street import BusyStreetBuilding, BusyStreetProp
from benchmark.venue_meetup.city_block import (
    BlockSide,
    CityBlockPlan,
    CityBlockPortal,
    DEFAULT_BLOCK_SIDE_LENGTH_CM,
    plan_city_block,
    plan_city_block_props,
)
from benchmark.venue_meetup.city_block_layout import OUTER_WALK_OFFSET_CM

Point2D = tuple[float, float]
DEFAULT_BLOCK_GAP_CM = 2_200.0


@dataclass(frozen=True, slots=True)
class ConnectedBlock:
    """One named block within the compact playtest district."""

    block_id: str
    display_name: str
    plan: CityBlockPlan


@dataclass(frozen=True, slots=True)
class AlleyLink:
    """A physical link between aligned outward portal nodes on two blocks."""

    alley_id: str
    first_block_id: str
    first_portal_side: BlockSide
    second_block_id: str
    second_portal_side: BlockSide
    start: Point2D
    end: Point2D
    clear_width_cm: float

    @property
    def length_cm(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True, slots=True)
class ConnectedBlocksPlan:
    """Three blocks and the two alleys that join their walk graphs."""

    center: Point2D
    block_gap_cm: float
    blocks: tuple[ConnectedBlock, ...]
    alleys: tuple[AlleyLink, ...]

    @property
    def buildings(self) -> tuple[BusyStreetBuilding, ...]:
        return tuple(
            sorted(
                (
                    building
                    for block in self.blocks
                    for building in block.plan.buildings
                ),
                key=lambda building: building.placement.index,
            )
        )

    def block_by_id(self, block_id: str) -> ConnectedBlock:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise ValueError(f"Unknown connected block id: {block_id}")

    def block_for_building(self, building_index: int) -> ConnectedBlock:
        for block in self.blocks:
            if any(
                building.placement.index == building_index
                for building in block.plan.buildings
            ):
                return block
        raise ValueError(f"Unknown connected-block building index: {building_index}")


def _alley(
    alley_id: str,
    first: ConnectedBlock,
    first_side: BlockSide,
    second: ConnectedBlock,
    second_side: BlockSide,
) -> AlleyLink:
    first_portal: CityBlockPortal = first.plan.portal_by_side(first_side)
    second_portal: CityBlockPortal = second.plan.portal_by_side(second_side)
    start = first_portal.offset_position(OUTER_WALK_OFFSET_CM)
    end = second_portal.offset_position(OUTER_WALK_OFFSET_CM)
    if not math.isclose(start[1], end[1], abs_tol=1e-6):
        raise ValueError(f"Alley {alley_id!r} portals must be horizontally aligned")
    return AlleyLink(
        alley_id=alley_id,
        first_block_id=first.block_id,
        first_portal_side=first_side,
        second_block_id=second.block_id,
        second_portal_side=second_side,
        start=start,
        end=end,
        clear_width_cm=min(
            first_portal.conservative_clear_width_cm,
            second_portal.conservative_clear_width_cm,
        ),
    )


def plan_connected_blocks(
    *,
    center: Point2D = (0.0, 0.0),
    block_gap_cm: float = DEFAULT_BLOCK_GAP_CM,
) -> ConnectedBlocksPlan:
    """Return west, central, and east blocks connected in one dense row."""

    center = (float(center[0]), float(center[1]))
    if not all(math.isfinite(value) for value in (*center, block_gap_cm)):
        raise ValueError("Connected-block center and gap must be finite")
    if block_gap_cm <= 2.0 * OUTER_WALK_OFFSET_CM:
        raise ValueError(
            "block_gap_cm must leave positive space between outer sidewalk rings"
        )
    pitch = DEFAULT_BLOCK_SIDE_LENGTH_CM + float(block_gap_cm)
    specs = (
        ("west", "West Market Block", (center[0] - pitch, center[1]), 0),
        ("central", "Central Arcade Block", center, 1),
        ("east", "East Tower Block", (center[0] + pitch, center[1]), 2),
    )
    blocks = tuple(
        ConnectedBlock(
            block_id=block_id,
            display_name=display_name,
            plan=plan_city_block(center=block_center, module_cycle=module_cycle),
        )
        for block_id, display_name, block_center, module_cycle in specs
    )
    west, central, east = blocks
    alleys = (
        _alley("west_central_alley", west, "east", central, "west"),
        _alley("central_east_alley", central, "east", east, "west"),
    )
    if any(alley.length_cm <= 0.0 for alley in alleys):
        raise ValueError("Connected-block alleys must have positive length")
    return ConnectedBlocksPlan(
        center=center,
        block_gap_cm=float(block_gap_cm),
        blocks=blocks,
        alleys=alleys,
    )


def plan_connected_block_props(
    plan: ConnectedBlocksPlan,
) -> tuple[BusyStreetProp, ...]:
    """Flatten facade props with globally unique report indices."""

    props: list[BusyStreetProp] = []
    for block in plan.blocks:
        first_index = len(props)
        block_props = plan_city_block_props(block.plan)
        props.extend(
            replace(prop, index=first_index + local_index)
            for local_index, prop in enumerate(block_props)
        )
    return tuple(props)


__all__ = [
    "AlleyLink",
    "ConnectedBlock",
    "ConnectedBlocksPlan",
    "DEFAULT_BLOCK_GAP_CM",
    "plan_connected_block_props",
    "plan_connected_blocks",
]
