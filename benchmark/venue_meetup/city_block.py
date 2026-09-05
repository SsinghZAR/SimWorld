"""Measured four-sided city-block composition for the playtest district.

The block reuses the authored busy-street facade set as eight independent
three-building runs. Two runs meet at every corner and stop around one centred
portal on each side, producing a continuous street wall with exactly four
intentional openings into the courtyard.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from typing import Literal

from benchmark.venue_meetup.busy_street import (
    BUSY_STREET_MODULES,
    BusyStreetBuilding,
    BusyStreetProp,
    plan_busy_street_modules,
    plan_busy_street_props,
)

Point2D = tuple[float, float]
BlockSide = Literal["north", "east", "south", "west"]

DEFAULT_BLOCK_SIDE_LENGTH_CM = 7_200.0
DEFAULT_PORTAL_WIDTH_CM = 1_200.0
DEFAULT_FACADE_FILL_RATIO = 0.78
DEFAULT_SETBACK_CM = 0.0

SIDE_OUTWARD: dict[BlockSide, Point2D] = {
    "north": (0.0, 1.0),
    "east": (1.0, 0.0),
    "south": (0.0, -1.0),
    "west": (-1.0, 0.0),
}


@dataclass(frozen=True, slots=True)
class CityBlockRun:
    """One measured half-side between a corner and a portal."""

    run_id: str
    side: BlockSide
    start: Point2D
    end: Point2D
    outward: Point2D
    module_indices: tuple[int, ...]
    buildings: tuple[BusyStreetBuilding, ...]

    @property
    def length_cm(self) -> float:
        return math.dist(self.start, self.end)


@dataclass(frozen=True, slots=True)
class CityBlockPortal:
    """One centred opening through a side of the perimeter wall."""

    portal_id: str
    side: BlockSide
    boundary_position: Point2D
    outward: Point2D
    planned_width_cm: float
    conservative_clear_width_cm: float

    def offset_position(self, offset_cm: float) -> Point2D:
        """Return a point offset from the wall; positive values point outside."""

        return (
            self.boundary_position[0] + self.outward[0] * float(offset_cm),
            self.boundary_position[1] + self.outward[1] * float(offset_cm),
        )


@dataclass(frozen=True, slots=True)
class CityBlockPlan:
    """Complete perimeter geometry with four portals and stable facade ids."""

    center: Point2D
    side_length_cm: float
    portal_width_cm: float
    facade_fill_ratio: float
    setback_cm: float
    runs: tuple[CityBlockRun, ...]
    portals: tuple[CityBlockPortal, ...]

    @property
    def half_extent_cm(self) -> float:
        return self.side_length_cm / 2.0

    @property
    def buildings(self) -> tuple[BusyStreetBuilding, ...]:
        return tuple(
            sorted(
                (building for run in self.runs for building in run.buildings),
                key=lambda building: building.placement.index,
            )
        )

    def buildings_on_side(
        self, side: BlockSide
    ) -> tuple[BusyStreetBuilding, ...]:
        return tuple(
            building
            for run in self.runs
            if run.side == side
            for building in run.buildings
        )

    def side_for_building(self, building_index: int) -> BlockSide:
        for run in self.runs:
            if any(
                building.placement.index == building_index
                for building in run.buildings
            ):
                return run.side
        raise ValueError(f"Unknown city-block building index: {building_index}")

    def portal_by_side(self, side: BlockSide) -> CityBlockPortal:
        return next(portal for portal in self.portals if portal.side == side)


# The endpoints put visually deep tower assets in the middle of their runs,
# rather than at a portal or corner. This keeps all four collision-envelope
# clearances generous while retaining the complete authored module set.
_RUN_MODULES: tuple[tuple[int, int, int], ...] = (
    (0, 1, 2),
    (3, 4, 5),
    (6, 8, 7),
    (9, 10, 11),
    (12, 13, 14),
    (15, 16, 17),
    (19, 18, 20),
    (21, 22, 23),
)


def _translated(center: Point2D, point: Point2D) -> Point2D:
    return center[0] + point[0], center[1] + point[1]


def _portal_clear_width(
    runs: tuple[CityBlockRun, ...],
    *,
    side: BlockSide,
    center: Point2D,
) -> float:
    """Return physical clear width after measured asset-bound overhangs."""

    outward = SIDE_OUTWARD[side]
    tangent = (-outward[1], outward[0])
    intervals: list[tuple[float, float, float]] = []
    for run in runs:
        if run.side != side:
            continue
        for building in run.buildings:
            placement = building.placement
            along = (
                (placement.position[0] - center[0]) * tangent[0]
                + (placement.position[1] - center[1]) * tangent[1]
            )
            half_width = placement.measured_tangent_width_cm / 2.0
            intervals.append((along, along - half_width, along + half_width))

    negative_edges = [end for along, _start, end in intervals if along < 0.0]
    positive_edges = [start for along, start, _end in intervals if along > 0.0]
    if not negative_edges or not positive_edges:
        raise ValueError(f"Side {side!r} does not bracket its centred portal")
    return min(positive_edges) - max(negative_edges)


def plan_city_block(
    *,
    center: Point2D = (0.0, 0.0),
    side_length_cm: float = DEFAULT_BLOCK_SIDE_LENGTH_CM,
    portal_width_cm: float = DEFAULT_PORTAL_WIDTH_CM,
    facade_fill_ratio: float = DEFAULT_FACADE_FILL_RATIO,
    setback_cm: float = DEFAULT_SETBACK_CM,
) -> CityBlockPlan:
    """Pack all 24 facades around a square wall with four centred portals."""

    center = (float(center[0]), float(center[1]))
    values = (
        *center,
        float(side_length_cm),
        float(portal_width_cm),
        float(facade_fill_ratio),
        float(setback_cm),
    )
    if not all(math.isfinite(value) for value in values):
        raise ValueError("City-block dimensions and center must be finite")
    if side_length_cm <= 0.0 or not 0.0 < portal_width_cm < side_length_cm:
        raise ValueError("portal_width_cm must be positive and smaller than the block")
    if not 0.0 < facade_fill_ratio <= 1.0 or setback_cm < 0.0:
        raise ValueError("Facade fill must be in (0, 1] and setback non-negative")

    half = float(side_length_cm) / 2.0
    portal_half = float(portal_width_cm) / 2.0
    local_segments: tuple[
        tuple[str, BlockSide, Point2D, Point2D, Point2D], ...
    ] = (
        ("north_west", "north", (-half, half), (-portal_half, half), (0.0, 1.0)),
        ("north_east", "north", (portal_half, half), (half, half), (0.0, 1.0)),
        ("east_north", "east", (half, half), (half, portal_half), (1.0, 0.0)),
        ("east_south", "east", (half, -portal_half), (half, -half), (1.0, 0.0)),
        ("south_east", "south", (half, -half), (portal_half, -half), (0.0, -1.0)),
        ("south_west", "south", (-portal_half, -half), (-half, -half), (0.0, -1.0)),
        ("west_south", "west", (-half, -portal_half), (-half, -half), (-1.0, 0.0)),
        ("west_north", "west", (-half, portal_half), (-half, half), (-1.0, 0.0)),
    )

    runs: list[CityBlockRun] = []
    for segment, module_indices in zip(local_segments, _RUN_MODULES):
        run_id, side, local_start, local_end, outward = segment
        start = _translated(center, local_start)
        end = _translated(center, local_end)
        buildings = plan_busy_street_modules(
            start,
            end,
            outward=outward,
            module_indices=module_indices,
            facade_fill_ratio=facade_fill_ratio,
            setback_cm=setback_cm,
        )
        runs.append(
            CityBlockRun(
                run_id=run_id,
                side=side,
                start=start,
                end=end,
                outward=outward,
                module_indices=module_indices,
                buildings=buildings,
            )
        )

    packed_runs = tuple(runs)
    if sorted(
        building.placement.index
        for run in packed_runs
        for building in run.buildings
    ) != list(range(len(BUSY_STREET_MODULES))):
        raise RuntimeError("City-block run specification must use every module once")

    boundary_positions: dict[BlockSide, Point2D] = {
        "north": _translated(center, (0.0, half)),
        "east": _translated(center, (half, 0.0)),
        "south": _translated(center, (0.0, -half)),
        "west": _translated(center, (-half, 0.0)),
    }
    portals = tuple(
        CityBlockPortal(
            portal_id=f"portal_{side}",
            side=side,
            boundary_position=boundary_positions[side],
            outward=SIDE_OUTWARD[side],
            planned_width_cm=float(portal_width_cm),
            conservative_clear_width_cm=_portal_clear_width(
                packed_runs,
                side=side,
                center=center,
            ),
        )
        for side in ("north", "east", "south", "west")
    )
    if any(portal.conservative_clear_width_cm <= 0.0 for portal in portals):
        raise ValueError("Measured building envelopes close at least one portal")

    return CityBlockPlan(
        center=center,
        side_length_cm=float(side_length_cm),
        portal_width_cm=float(portal_width_cm),
        facade_fill_ratio=float(facade_fill_ratio),
        setback_cm=float(setback_cm),
        runs=packed_runs,
        portals=portals,
    )


def plan_city_block_props(plan: CityBlockPlan) -> tuple[BusyStreetProp, ...]:
    """Place facade cues per run while keeping all four portals unobstructed."""

    props: list[BusyStreetProp] = []
    for run in plan.runs:
        run_props = plan_busy_street_props(
            run.buildings,
            run.start,
            run.end,
            outward=run.outward,
            setback_cm=plan.setback_cm,
        )
        first_index = len(props)
        props.extend(
            replace(prop, index=first_index + local_index)
            for local_index, prop in enumerate(run_props)
        )
    return tuple(props)


__all__ = [
    "BlockSide",
    "CityBlockPlan",
    "CityBlockPortal",
    "CityBlockRun",
    "DEFAULT_BLOCK_SIDE_LENGTH_CM",
    "DEFAULT_FACADE_FILL_RATIO",
    "DEFAULT_PORTAL_WIDTH_CM",
    "DEFAULT_SETBACK_CM",
    "SIDE_OUTWARD",
    "plan_city_block",
    "plan_city_block_props",
]
