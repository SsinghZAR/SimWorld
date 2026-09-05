"""Pure plans for scalable Rosebank-inspired mixed-use districts.

The plan borrows Rosebank's legible ingredients rather than copying its cadastral
map: a strong transit avenue, retail high streets, quieter residential edges,
landmark-led wayfinding, green pockets, and mid-block pedestrian/service alleys.
Coordinates use Unreal centimetres.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from benchmark.venue_meetup.scenario import VenueType

Point2D = tuple[float, float]
BlockSide = Literal["north", "east", "south", "west"]
BlockZone = Literal[
    "transit_core",
    "mixed_use",
    "residential",
    "civic",
    "garden",
]
AlleyAxis = Literal["horizontal", "vertical"]

GRID_SIZE = 9
SUPPORTED_GRID_SIZES = (3, 5, 7, 9)
ROSEBANK_GRID_TEMPLATE_IDS = {
    3: "rosebank_grid_3x3_v0",
    5: "rosebank_grid_5x5_v0",
    7: "rosebank_grid_7x7_v0",
    9: "rosebank_grid_9x9_v0",
}
ROSEBANK_GRID_VENUE_COUNTS = {3: 4, 5: 8, 7: 12, 9: 36}
ROSEBANK_GRID_MAX_STEPS = {3: 96, 5: 160, 7: 256, 9: 384}
BLOCK_SIDE_CM = 5_600.0
BLOCK_PITCH_CM = 7_600.0
STREET_GAP_CM = BLOCK_PITCH_CM - BLOCK_SIDE_CM
MINOR_STREET_WIDTH_CM = 1_200.0
SECONDARY_STREET_WIDTH_CM = 1_500.0
OXFORD_ROAD_WIDTH_CM = 2_000.0
HIGH_STREET_WIDTH_CM = 1_700.0
ALLEY_WIDTH_CM = 900.0
ALLEY_FRONTAGE_OFFSET_CM = 1_500.0
SIDEWALK_WIDTH_CM = 400.0

VERTICAL_STREET_NAMES = (
    "west_boundary_road",
    "jan_smuts_link",
    "keyes_avenue",
    "art_lane",
    "cradock_avenue",
    "oxford_road",
    "bath_avenue",
    "sturdee_avenue",
    "glenhove_link",
    "east_boundary_road",
)
HORIZONTAL_STREET_NAMES = (
    "south_boundary_road",
    "bolton_road",
    "jellicoe_avenue",
    "baker_street",
    "the_zone_walk",
    "tyrwhitt_high_street",
    "rosebank_road",
    "north_link",
    "parks_edge_road",
    "north_boundary_road",
)

_VERTICAL_STREET_NAMES_BY_SIZE = {
    3: (
        "west_boundary_road",
        "cradock_avenue",
        "oxford_road",
        "east_boundary_road",
    ),
    5: (
        "west_boundary_road",
        "keyes_avenue",
        "cradock_avenue",
        "oxford_road",
        "bath_avenue",
        "east_boundary_road",
    ),
    7: (
        "west_boundary_road",
        "jan_smuts_link",
        "keyes_avenue",
        "cradock_avenue",
        "oxford_road",
        "bath_avenue",
        "sturdee_avenue",
        "east_boundary_road",
    ),
    9: VERTICAL_STREET_NAMES,
}
_HORIZONTAL_STREET_NAMES_BY_SIZE = {
    3: (
        "south_boundary_road",
        "the_zone_walk",
        "tyrwhitt_high_street",
        "north_boundary_road",
    ),
    5: (
        "south_boundary_road",
        "jellicoe_avenue",
        "baker_street",
        "tyrwhitt_high_street",
        "rosebank_road",
        "north_boundary_road",
    ),
    7: (
        "south_boundary_road",
        "bolton_road",
        "jellicoe_avenue",
        "baker_street",
        "tyrwhitt_high_street",
        "rosebank_road",
        "north_link",
        "north_boundary_road",
    ),
    9: HORIZONTAL_STREET_NAMES,
}

_LANDMARK_COORDINATES_BY_SIZE: dict[
    int,
    tuple[tuple[int, int, str], ...],
] = {
    3: (
        (0, 1, "clock_tower"),
        (1, 1, "gautrain_tower"),
        (2, 1, "civic_hall"),
    ),
    5: (
        (1, 0, "clock_tower"),
        (3, 1, "arts_centre"),
        (2, 1, "market_hall"),
        (2, 2, "gautrain_tower"),
        (3, 3, "hotel_tower"),
        (1, 4, "civic_hall"),
    ),
    7: (
        (2, 1, "clock_tower"),
        (5, 2, "arts_centre"),
        (3, 2, "market_hall"),
        (3, 3, "gautrain_tower"),
        (4, 4, "hotel_tower"),
        (2, 5, "civic_hall"),
    ),
    9: (
        (3, 1, "clock_tower"),
        (6, 2, "arts_centre"),
        (4, 3, "market_hall"),
        (4, 4, "gautrain_tower"),
        (5, 5, "hotel_tower"),
        (2, 6, "civic_hall"),
    ),
}
_GARDEN_COORDINATES_BY_SIZE: dict[int, tuple[tuple[int, int], ...]] = {
    3: (),
    5: ((0, 0), (0, 4), (4, 0), (4, 4)),
    7: ((1, 1), (1, 5), (5, 1), (5, 5)),
    9: ((1, 1), (1, 7), (7, 1), (7, 7)),
}

_VENUE_TYPES: tuple[VenueType, ...] = (
    "restaurant",
    "cafe",
    "shop",
    "bookshop",
    "bar",
    "hotel_lobby",
    "skyscraper_lobby",
    "pub",
)
_VENUE_ASSETS: dict[VenueType, tuple[str, tuple[float, float, float]]] = {
    "restaurant": ("BP_Building_05_C", (0.42, 0.42, 0.55)),
    "cafe": ("BP_Building_06_C", (0.40, 0.40, 0.52)),
    "shop": ("BP_Building_25_C", (0.30, 0.30, 0.42)),
    "bookshop": ("BP_Building_24_C", (0.34, 0.34, 0.46)),
    "bar": ("BP_Building_44_C", (0.28, 0.28, 0.42)),
    "hotel_lobby": ("BP_Building_95_C", (0.17, 0.17, 0.28)),
    "skyscraper_lobby": ("BP_Building_101_C", (0.22, 0.22, 0.50)),
    "pub": ("BP_Building_44_C", (0.28, 0.28, 0.38)),
}
_SIDE_TANGENTS: dict[BlockSide, Point2D] = {
    "north": (1.0, 0.0),
    "east": (0.0, -1.0),
    "south": (-1.0, 0.0),
    "west": (0.0, 1.0),
}


@dataclass(frozen=True, slots=True)
class RosebankBlock:
    """One addressable cell in a scalable district."""

    row: int
    column: int
    block_id: str
    center: Point2D
    zone: BlockZone
    visual_style: str
    alley_axes: frozenset[AlleyAxis]
    landmark_role: str | None = None


@dataclass(frozen=True, slots=True)
class RosebankVenueSite:
    """One public-facing venue attached to a mixed-use block edge."""

    venue_id: str
    slot_id: str
    display_name: str
    block_id: str
    side: BlockSide
    venue_type: VenueType
    asset_key: str
    scale: tuple[float, float, float]
    zone_id: str
    frontage_offset_cm: float = 0.0


@dataclass(frozen=True, slots=True)
class RosebankGridPlan:
    """Complete district plan shared by layout, massing, and scenario builders."""

    grid_size: int
    center: Point2D
    blocks: tuple[RosebankBlock, ...]
    venue_sites: tuple[RosebankVenueSite, ...]
    street_x: tuple[float, ...]
    street_y: tuple[float, ...]
    vertical_street_names: tuple[str, ...]
    horizontal_street_names: tuple[str, ...]
    primary_street_index: int

    def block_by_id(self, block_id: str) -> RosebankBlock:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise ValueError(f"Unknown Rosebank block id: {block_id}")

    def block_at(self, row: int, column: int) -> RosebankBlock:
        if not 0 <= row < self.grid_size or not 0 <= column < self.grid_size:
            raise ValueError(
                f"Block coordinate outside {self.grid_size}x{self.grid_size}: "
                f"{(row, column)}"
            )
        return self.blocks[row * self.grid_size + column]

    @property
    def extent_cm(self) -> float:
        return max(abs(value) for value in (*self.street_x, *self.street_y))


def block_id(row: int, column: int, *, grid_size: int = GRID_SIZE) -> str:
    """Return the public letter-number address for a grid cell."""

    if not 0 <= row < grid_size or not 0 <= column < grid_size:
        raise ValueError(
            f"Block coordinate outside {grid_size}x{grid_size}: {(row, column)}"
        )
    return f"{chr(ord('A') + column)}{row + 1}"


def _landmark_block_roles(grid_size: int) -> dict[str, str]:
    return {
        block_id(row, column, grid_size=grid_size): role
        for row, column, role in _LANDMARK_COORDINATES_BY_SIZE[grid_size]
    }


def _garden_block_ids(grid_size: int) -> frozenset[str]:
    return frozenset(
        block_id(row, column, grid_size=grid_size)
        for row, column in _GARDEN_COORDINATES_BY_SIZE[grid_size]
    )


# Compatibility aliases for callers that describe the original 9x9 map.
LANDMARK_BLOCK_ROLES: dict[str, str] = _landmark_block_roles(GRID_SIZE)
GARDEN_BLOCK_IDS = _garden_block_ids(GRID_SIZE)


def frontage_tangent(side: BlockSide) -> Point2D:
    """Return the clockwise unit tangent for one block frontage."""

    return _SIDE_TANGENTS[side]


def _block_zone(
    row: int,
    column: int,
    label: str,
    *,
    grid_size: int,
    landmark_roles: dict[str, str],
    garden_ids: frozenset[str],
) -> BlockZone:
    if label in garden_ids:
        return "garden"
    if label in landmark_roles:
        return (
            "transit_core"
            if landmark_roles[label] == "gautrain_tower"
            else "civic"
        )
    center_index = grid_size // 2
    distance = max(abs(row - center_index), abs(column - center_index))
    core_radius = 0 if grid_size == 3 else 1
    mixed_radius = max(1, grid_size // 4)
    if distance <= core_radius:
        return "transit_core"
    if (
        abs(row - center_index) <= mixed_radius
        or abs(column - center_index) <= mixed_radius
    ):
        return "mixed_use"
    return "residential"


def _visual_style(zone: BlockZone) -> str:
    return {
        "transit_core": "rosebank_core",
        "mixed_use": "rosebank_mixed",
        "residential": "rosebank_residential",
        "civic": "rosebank_civic",
        "garden": "rosebank_garden",
    }[zone]


def _alley_blocks(
    grid_size: int,
    *,
    landmark_roles: dict[str, str],
    garden_ids: frozenset[str],
) -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
    if grid_size == GRID_SIZE:
        horizontal = {
            (row, column)
            for row in (2, 4, 6)
            for column in range(1, 8)
        }
        vertical = {
            (row, column)
            for column in (2, 6)
            for row in range(1, 8)
        }
    elif grid_size == 3:
        horizontal = {(1, 0), (1, 2)}
        vertical = {(0, 0), (2, 2)}
    else:
        offsets = tuple(range(2, grid_size - 1, 2))
        interior = range(1, grid_size - 1)
        horizontal = {
            (row, column) for row in offsets for column in interior
        }
        vertical = {
            (row, column) for column in offsets for row in interior
        }
    for label in garden_ids:
        column = ord(label[0]) - ord("A")
        row = int(label[1:]) - 1
        horizontal.add((row, column))
        vertical.add((row, column))
    landmark_coordinates = {
        (int(label[1:]) - 1, ord(label[0]) - ord("A"))
        for label in landmark_roles
    }
    return horizontal - landmark_coordinates, vertical - landmark_coordinates


def _venue_sites(
    blocks: tuple[RosebankBlock, ...],
    *,
    grid_size: int,
    venue_count: int,
    landmark_roles: dict[str, str],
    garden_ids: frozenset[str],
) -> tuple[RosebankVenueSite, ...]:
    if venue_count < 4 or venue_count % 2:
        raise ValueError(
            f"Rosebank grids require an even venue count of at least 4, got {venue_count}"
        )
    excluded = set(landmark_roles) | set(garden_ids)
    center_index = grid_size // 2
    venues_per_zone = venue_count // 2

    def candidates(*, west: bool) -> list[RosebankBlock]:
        selected = [
            block
            for block in blocks
            if block.block_id not in excluded
            and (
                block.column < center_index
                if west
                else block.column > center_index
            )
        ]
        ranked = sorted(
            selected,
            key=lambda block: (
                abs(block.row - center_index)
                + abs(block.column - center_index),
                abs(block.row - center_index),
                block.row,
                block.column,
            ),
        )
        if len(ranked) < venues_per_zone:
            side = "west" if west else "east"
            raise ValueError(
                f"{grid_size}x{grid_size} grid only has {len(ranked)} usable "
                f"{side}-zone blocks for {venues_per_zone} venues"
            )
        return ranked[:venues_per_zone]

    sites: list[RosebankVenueSite] = []
    for zone_id, selected in (
        ("zone_west", candidates(west=True)),
        ("zone_east", candidates(west=False)),
    ):
        for block in selected:
            venue_type = _VENUE_TYPES[len(sites) % len(_VENUE_TYPES)]
            asset_key, scale = _VENUE_ASSETS[venue_type]
            side: BlockSide = ("north", "east", "south", "west")[
                (block.row * 3 + block.column) % 4
            ]
            crosses_alley_opening = (
                "vertical" in block.alley_axes
                and side in {"north", "south"}
            ) or (
                "horizontal" in block.alley_axes
                and side in {"east", "west"}
            )
            frontage_offset_cm = 0.0
            if crosses_alley_opening:
                direction = 1.0 if (block.row + block.column) % 2 else -1.0
                frontage_offset_cm = direction * ALLEY_FRONTAGE_OFFSET_CM
            if not sites:
                slug = "red_awning_bistro"
                display_name = "Rosebank Red Awning Bistro"
            else:
                slug = f"{block.block_id.lower()}_{venue_type}"
                display_name = (
                    f"{block.block_id} {venue_type.replace('_', ' ').title()}"
                )
            sites.append(
                RosebankVenueSite(
                    venue_id=f"venue_{slug}",
                    slot_id=slug,
                    display_name=display_name,
                    block_id=block.block_id,
                    side=side,
                    venue_type=venue_type,
                    asset_key=asset_key,
                    scale=scale,
                    zone_id=zone_id,
                    frontage_offset_cm=frontage_offset_cm,
                )
            )
    return tuple(sites)


def plan_rosebank_grid(
    *,
    center: Point2D = (0.0, 0.0),
    grid_size: int = GRID_SIZE,
    venue_count: int | None = None,
) -> RosebankGridPlan:
    """Return a deterministic, supported odd-sized mixed-use district plan."""

    if grid_size not in SUPPORTED_GRID_SIZES:
        raise ValueError(
            f"Unsupported Rosebank grid size {grid_size}; expected one of "
            f"{SUPPORTED_GRID_SIZES}"
        )
    resolved_venue_count = (
        ROSEBANK_GRID_VENUE_COUNTS[grid_size]
        if venue_count is None
        else int(venue_count)
    )
    center = float(center[0]), float(center[1])
    half_grid = grid_size / 2.0
    street_x = tuple(
        center[0] + (index - half_grid) * BLOCK_PITCH_CM
        for index in range(grid_size + 1)
    )
    street_y = tuple(
        center[1] + (index - half_grid) * BLOCK_PITCH_CM
        for index in range(grid_size + 1)
    )
    landmark_roles = _landmark_block_roles(grid_size)
    garden_ids = _garden_block_ids(grid_size)
    horizontal_alleys, vertical_alleys = _alley_blocks(
        grid_size,
        landmark_roles=landmark_roles,
        garden_ids=garden_ids,
    )
    center_index = grid_size // 2
    blocks: list[RosebankBlock] = []
    for row in range(grid_size):
        for column in range(grid_size):
            label = block_id(row, column, grid_size=grid_size)
            zone = _block_zone(
                row,
                column,
                label,
                grid_size=grid_size,
                landmark_roles=landmark_roles,
                garden_ids=garden_ids,
            )
            axes: set[AlleyAxis] = set()
            if (row, column) in horizontal_alleys:
                axes.add("horizontal")
            if (row, column) in vertical_alleys:
                axes.add("vertical")
            blocks.append(
                RosebankBlock(
                    row=row,
                    column=column,
                    block_id=label,
                    center=(
                        center[0]
                        + (column - center_index) * BLOCK_PITCH_CM,
                        center[1] + (row - center_index) * BLOCK_PITCH_CM,
                    ),
                    zone=zone,
                    visual_style=_visual_style(zone),
                    alley_axes=frozenset(axes),
                    landmark_role=landmark_roles.get(label),
                )
            )
    block_tuple = tuple(blocks)
    return RosebankGridPlan(
        grid_size=grid_size,
        center=center,
        blocks=block_tuple,
        venue_sites=_venue_sites(
            block_tuple,
            grid_size=grid_size,
            venue_count=resolved_venue_count,
            landmark_roles=landmark_roles,
            garden_ids=garden_ids,
        ),
        street_x=street_x,
        street_y=street_y,
        vertical_street_names=_VERTICAL_STREET_NAMES_BY_SIZE[grid_size],
        horizontal_street_names=_HORIZONTAL_STREET_NAMES_BY_SIZE[grid_size],
        primary_street_index=grid_size // 2 + 1,
    )


__all__ = [
    "ALLEY_WIDTH_CM",
    "ALLEY_FRONTAGE_OFFSET_CM",
    "BLOCK_PITCH_CM",
    "BLOCK_SIDE_CM",
    "BlockSide",
    "BlockZone",
    "GARDEN_BLOCK_IDS",
    "GRID_SIZE",
    "HIGH_STREET_WIDTH_CM",
    "HORIZONTAL_STREET_NAMES",
    "LANDMARK_BLOCK_ROLES",
    "MINOR_STREET_WIDTH_CM",
    "OXFORD_ROAD_WIDTH_CM",
    "ROSEBANK_GRID_MAX_STEPS",
    "ROSEBANK_GRID_TEMPLATE_IDS",
    "ROSEBANK_GRID_VENUE_COUNTS",
    "RosebankBlock",
    "RosebankGridPlan",
    "RosebankVenueSite",
    "SECONDARY_STREET_WIDTH_CM",
    "SIDEWALK_WIDTH_CM",
    "STREET_GAP_CM",
    "SUPPORTED_GRID_SIZES",
    "VERTICAL_STREET_NAMES",
    "block_id",
    "frontage_tangent",
    "plan_rosebank_grid",
]
