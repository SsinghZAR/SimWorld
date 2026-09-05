"""Pure plan for a Rosebank-inspired nine-by-nine mixed-use district.

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

LANDMARK_BLOCK_ROLES: dict[str, str] = {
    "B4": "clock_tower",
    "C7": "arts_centre",
    "D5": "market_hall",
    "E5": "gautrain_tower",
    "F6": "hotel_tower",
    "G3": "civic_hall",
}
GARDEN_BLOCK_IDS = frozenset({"B2", "H2", "B8", "H8"})

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
    """One addressable cell in the nine-by-nine district."""

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

    center: Point2D
    blocks: tuple[RosebankBlock, ...]
    venue_sites: tuple[RosebankVenueSite, ...]
    street_x: tuple[float, ...]
    street_y: tuple[float, ...]

    def block_by_id(self, block_id: str) -> RosebankBlock:
        for block in self.blocks:
            if block.block_id == block_id:
                return block
        raise ValueError(f"Unknown Rosebank block id: {block_id}")

    def block_at(self, row: int, column: int) -> RosebankBlock:
        if not 0 <= row < GRID_SIZE or not 0 <= column < GRID_SIZE:
            raise ValueError(f"Block coordinate outside {GRID_SIZE}x{GRID_SIZE}: {(row, column)}")
        return self.blocks[row * GRID_SIZE + column]

    @property
    def extent_cm(self) -> float:
        return max(abs(value) for value in (*self.street_x, *self.street_y))


def block_id(row: int, column: int) -> str:
    """Return the public A1-I9 address for a grid cell."""

    if not 0 <= row < GRID_SIZE or not 0 <= column < GRID_SIZE:
        raise ValueError(f"Block coordinate outside {GRID_SIZE}x{GRID_SIZE}: {(row, column)}")
    return f"{chr(ord('A') + column)}{row + 1}"


def frontage_tangent(side: BlockSide) -> Point2D:
    """Return the clockwise unit tangent for one block frontage."""

    return _SIDE_TANGENTS[side]


def _block_zone(row: int, column: int, label: str) -> BlockZone:
    if label in GARDEN_BLOCK_IDS:
        return "garden"
    if label in LANDMARK_BLOCK_ROLES:
        return "civic" if label != "E5" else "transit_core"
    distance = max(abs(row - 4), abs(column - 4))
    if distance <= 1:
        return "transit_core"
    if abs(row - 4) <= 2 or abs(column - 4) <= 2:
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


def _alley_blocks() -> tuple[set[tuple[int, int]], set[tuple[int, int]]]:
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
    for label in GARDEN_BLOCK_IDS:
        column = ord(label[0]) - ord("A")
        row = int(label[1:]) - 1
        horizontal.add((row, column))
        vertical.add((row, column))
    landmark_coordinates = {
        (int(label[1:]) - 1, ord(label[0]) - ord("A"))
        for label in LANDMARK_BLOCK_ROLES
    }
    return horizontal - landmark_coordinates, vertical - landmark_coordinates


def _venue_sites(blocks: tuple[RosebankBlock, ...]) -> tuple[RosebankVenueSite, ...]:
    excluded = set(LANDMARK_BLOCK_ROLES) | set(GARDEN_BLOCK_IDS)

    def candidates(*, west: bool) -> list[RosebankBlock]:
        selected = [
            block
            for block in blocks
            if block.block_id not in excluded
            and (block.column <= 3 if west else block.column >= 5)
        ]
        return sorted(
            selected,
            key=lambda block: (
                abs(block.row - 4) + abs(block.column - 4),
                abs(block.row - 4),
                block.row,
                block.column,
            ),
        )[:18]

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


def plan_rosebank_grid(*, center: Point2D = (0.0, 0.0)) -> RosebankGridPlan:
    """Return a deterministic 81-block mixed-use district plan."""

    center = float(center[0]), float(center[1])
    half_grid = GRID_SIZE / 2.0
    street_x = tuple(
        center[0] + (index - half_grid) * BLOCK_PITCH_CM
        for index in range(GRID_SIZE + 1)
    )
    street_y = tuple(
        center[1] + (index - half_grid) * BLOCK_PITCH_CM
        for index in range(GRID_SIZE + 1)
    )
    horizontal_alleys, vertical_alleys = _alley_blocks()
    blocks: list[RosebankBlock] = []
    for row in range(GRID_SIZE):
        for column in range(GRID_SIZE):
            label = block_id(row, column)
            zone = _block_zone(row, column, label)
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
                        center[0] + (column - 4) * BLOCK_PITCH_CM,
                        center[1] + (row - 4) * BLOCK_PITCH_CM,
                    ),
                    zone=zone,
                    visual_style=_visual_style(zone),
                    alley_axes=frozenset(axes),
                    landmark_role=LANDMARK_BLOCK_ROLES.get(label),
                )
            )
    block_tuple = tuple(blocks)
    return RosebankGridPlan(
        center=center,
        blocks=block_tuple,
        venue_sites=_venue_sites(block_tuple),
        street_x=street_x,
        street_y=street_y,
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
    "RosebankBlock",
    "RosebankGridPlan",
    "RosebankVenueSite",
    "SECONDARY_STREET_WIDTH_CM",
    "SIDEWALK_WIDTH_CM",
    "STREET_GAP_CM",
    "VERTICAL_STREET_NAMES",
    "block_id",
    "frontage_tangent",
    "plan_rosebank_grid",
]
