"""Composition of the continuous wall primitive into a mixed-use street row."""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, replace
from typing import Literal

from benchmark.venue_meetup.street_wall import (
    StreetWallPlacement,
    plan_street_wall,
)

FacadeUse = Literal[
    "restaurant",
    "house",
    "bookshop",
    "bar",
    "skyscraper_lobby",
]
StreetPropUse = Literal[
    "restaurant_seating",
    "book_display",
    "bar_seating",
    "street_furniture",
]


@dataclass(frozen=True, slots=True)
class StreetFacadeModule:
    """One authored module in the non-repeating playtest frontage cadence."""

    asset_key: str
    use: FacadeUse
    venue_slug: str | None = None
    display_name: str | None = None
    visual_cue: str = ""


# Twenty-four modules form one complete block. The long rhythm avoids visible
# tiling while producing exactly twelve interactable venues split evenly by the
# street midpoint: six restaurants, two bookshops, two bars, and two tower
# lobbies. Houses remain solid visual/navigation context rather than candidates.
BUSY_STREET_MODULES = (
    StreetFacadeModule(
        "BP_Building_06_C",
        "restaurant",
        "green_awning_bistro",
        "Green Awning Bistro",
        "green awning and brick storefront",
    ),
    StreetFacadeModule("BP_Building_03_C", "house"),
    StreetFacadeModule(
        "BP_Building_24_C",
        "bookshop",
        "red_page_books",
        "Red Page Books",
        "red awnings, curved balconies, and a book display table",
    ),
    StreetFacadeModule("BP_Building_01_C", "house"),
    StreetFacadeModule(
        "BP_Building_44_C",
        "bar",
        "lantern_bar",
        "Lantern Bar",
        "colorful signs and a bright corner entrance",
    ),
    StreetFacadeModule("BP_Building_04_C", "house"),
    StreetFacadeModule(
        "BP_Building_05_C",
        "restaurant",
        "copper_kettle",
        "Copper Kettle",
        "red awning and outdoor table",
    ),
    StreetFacadeModule("BP_Building_84_C", "house"),
    StreetFacadeModule(
        "BP_Building_121_C",
        "skyscraper_lobby",
        "aurora_tower_lobby",
        "Aurora Tower Lobby",
        "curved glass tower and colorful billboards",
    ),
    StreetFacadeModule("BP_Building_81_C", "house"),
    StreetFacadeModule(
        "BP_Building_06_C",
        "restaurant",
        "market_lane_kitchen",
        "Market Lane Kitchen",
        "green awning and bright table seating",
    ),
    StreetFacadeModule("BP_Building_03_C", "house"),
    StreetFacadeModule("BP_Building_04_C", "house"),
    StreetFacadeModule(
        "BP_Building_05_C",
        "restaurant",
        "red_awning_grill",
        "Red Awning Grill",
        "red awning and brick shopfront",
    ),
    StreetFacadeModule(
        "BP_Building_27_C",
        "bookshop",
        "lantern_books",
        "Lantern Books",
        "sign-covered shopfront and pavement display",
    ),
    StreetFacadeModule("BP_Building_81_C", "house"),
    StreetFacadeModule(
        "BP_Building_06_C",
        "restaurant",
        "corner_table_cafe",
        "Corner Table Cafe",
        "corner storefront and green awning",
    ),
    StreetFacadeModule("BP_Building_01_C", "house"),
    StreetFacadeModule(
        "BP_Building_90_C",
        "bar",
        "music_box_bar",
        "Music Box Bar",
        "music billboard and illuminated commercial entrance",
    ),
    StreetFacadeModule("BP_Building_84_C", "house"),
    StreetFacadeModule(
        "BP_Building_05_C",
        "restaurant",
        "brickhouse_diner",
        "Brickhouse Diner",
        "dark metal shopfront and red awning",
    ),
    StreetFacadeModule("BP_Building_03_C", "house"),
    StreetFacadeModule(
        "BP_Building_126_C",
        "skyscraper_lobby",
        "neon_spire_lobby",
        "Neon Spire Lobby",
        "sleek tower facade and colorful billboards",
    ),
    StreetFacadeModule("BP_Building_04_C", "house"),
)

BUSY_STREET_SEQUENCE = tuple(module.asset_key for module in BUSY_STREET_MODULES)
RESTAURANT_ASSETS = frozenset(
    module.asset_key for module in BUSY_STREET_MODULES if module.use == "restaurant"
)
HOUSE_ASSETS = frozenset(
    module.asset_key for module in BUSY_STREET_MODULES if module.use == "house"
)
BOOKSHOP_ASSETS = frozenset(
    module.asset_key for module in BUSY_STREET_MODULES if module.use == "bookshop"
)
BAR_ASSETS = frozenset(
    module.asset_key for module in BUSY_STREET_MODULES if module.use == "bar"
)
SKYSCRAPER_ASSETS = frozenset(
    module.asset_key
    for module in BUSY_STREET_MODULES
    if module.use == "skyscraper_lobby"
)

BUSY_STREET_SCALES = {
    "BP_Building_01_C": 0.42,
    "BP_Building_03_C": 0.40,
    "BP_Building_04_C": 0.40,
    "BP_Building_05_C": 0.42,
    "BP_Building_06_C": 0.39,
    "BP_Building_24_C": 0.39,
    "BP_Building_27_C": 0.32,
    "BP_Building_44_C": 0.30,
    "BP_Building_81_C": 0.34,
    "BP_Building_84_C": 0.34,
    "BP_Building_90_C": 0.18,
    "BP_Building_121_C": 0.22,
    "BP_Building_126_C": 0.22,
}

BUSY_STREET_HEIGHT_FACTORS = {
    "BP_Building_01_C": 1.25,
    "BP_Building_03_C": 1.20,
    "BP_Building_04_C": 1.22,
    "BP_Building_05_C": 1.20,
    "BP_Building_06_C": 1.25,
    "BP_Building_24_C": 1.18,
    "BP_Building_27_C": 1.18,
    "BP_Building_44_C": 1.10,
    "BP_Building_81_C": 1.12,
    "BP_Building_84_C": 1.10,
    "BP_Building_90_C": 0.72,
    "BP_Building_121_C": 1.30,
    "BP_Building_126_C": 1.35,
}

BUSY_STREET_VENUE_MASKS = (
    (230, 55, 70),
    (45, 170, 220),
    (245, 145, 35),
    (120, 75, 210),
    (30, 190, 120),
    (225, 80, 170),
    (250, 205, 45),
    (60, 110, 230),
    (200, 95, 45),
    (45, 205, 195),
    (155, 205, 55),
    (190, 60, 220),
)

BUSY_STREET_PROP_SCALES = {
    "BP_Table_C": (0.78, 0.78, 0.78),
    "BP_Table2_C": (0.78, 0.78, 0.78),
    "BP_Trash_bin_a_C": (0.68, 0.68, 0.68),
    "BP_Hydrant_C": (0.70, 0.70, 0.70),
}


@dataclass(frozen=True, slots=True)
class BusyStreetBuilding:
    """One packed facade plus its readable street use."""

    placement: StreetWallPlacement
    use: FacadeUse
    venue_id: str | None = None
    display_name: str | None = None
    visual_cue: str = ""
    mask_color_rgb: tuple[int, int, int] | None = None


@dataclass(frozen=True, slots=True)
class BusyStreetProp:
    """One inert frontage cue placed on the public side of the wall."""

    index: int
    building_index: int
    asset_key: str
    position: tuple[float, float, float]
    yaw_deg: float
    scale: tuple[float, float, float]
    use: StreetPropUse


def decorate_busy_street_placements(
    placements: Sequence[StreetWallPlacement],
    *,
    module_indices: Sequence[int] | None = None,
) -> tuple[BusyStreetBuilding, ...]:
    """Attach authored use, identity, and mask metadata to packed facades.

    ``module_indices`` lets callers pack selected portions of the authored
    cadence independently while retaining stable global building indices and
    venue identities. This is what allows the same facade set to form either
    one straight test wall or several sides of a city block.
    """

    packed = tuple(placements)
    authored_indices = (
        tuple(item.index for item in packed)
        if module_indices is None
        else tuple(module_indices)
    )
    if len(authored_indices) != len(packed):
        raise ValueError("module_indices must match the number of placements")
    if any(
        isinstance(index, bool) or not isinstance(index, int) or index < 0
        for index in authored_indices
    ):
        raise ValueError("module_indices must contain non-negative integers")

    venues_per_cycle = sum(
        module.venue_slug is not None for module in BUSY_STREET_MODULES
    )
    result: list[BusyStreetBuilding] = []
    for item, authored_index in zip(packed, authored_indices):
        module_index = authored_index % len(BUSY_STREET_MODULES)
        cycle = authored_index // len(BUSY_STREET_MODULES)
        module = BUSY_STREET_MODULES[module_index]
        if item.asset_key != module.asset_key:
            raise ValueError(
                f"Placement asset {item.asset_key!r} does not match authored "
                f"module {authored_index} asset {module.asset_key!r}"
            )

        venue_id = None
        display_name = module.display_name
        mask_color = None
        if module.venue_slug is not None:
            suffix = f"_{cycle + 1}" if cycle else ""
            venue_id = f"venue_{module.venue_slug}{suffix}"
            if cycle and display_name is not None:
                display_name = f"{display_name} {cycle + 1}"
            venue_offset = sum(
                candidate.venue_slug is not None
                for candidate in BUSY_STREET_MODULES[:module_index]
            )
            venue_ordinal = cycle * venues_per_cycle + venue_offset
            mask_color = BUSY_STREET_VENUE_MASKS[
                venue_ordinal % len(BUSY_STREET_VENUE_MASKS)
            ]

        result.append(
            BusyStreetBuilding(
                placement=replace(item, index=authored_index),
                use=module.use,
                venue_id=venue_id,
                display_name=display_name,
                visual_cue=module.visual_cue,
                mask_color_rgb=mask_color,
            )
        )
    return tuple(result)


def plan_busy_street_modules(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    outward: tuple[float, float],
    module_indices: Sequence[int],
    facade_fill_ratio: float = 0.78,
    setback_cm: float = 200.0,
) -> tuple[BusyStreetBuilding, ...]:
    """Pack a selected authored module run between two boundary points."""

    indices = tuple(module_indices)
    if not indices:
        raise ValueError("module_indices must not be empty")
    modules = tuple(
        BUSY_STREET_MODULES[index % len(BUSY_STREET_MODULES)]
        for index in indices
    )
    placements = plan_street_wall(
        start,
        end,
        outward=outward,
        asset_keys=tuple(module.asset_key for module in modules),
        preferred_scales=BUSY_STREET_SCALES,
        gap_cm=0.0,
        setback_cm=setback_cm,
        facade_fill_ratio=facade_fill_ratio,
        height_factors=BUSY_STREET_HEIGHT_FACTORS,
        target_count=len(modules),
    )
    return decorate_busy_street_placements(
        placements,
        module_indices=indices,
    )


def plan_busy_street(
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    outward: tuple[float, float],
    facade_fill_ratio: float = 0.78,
    setback_cm: float = 200.0,
    target_count: int | None = len(BUSY_STREET_MODULES),
) -> tuple[BusyStreetBuilding, ...]:
    """Return the zero-gap authored mixed-use playtest frontage."""

    placements = plan_street_wall(
        start,
        end,
        outward=outward,
        asset_keys=BUSY_STREET_SEQUENCE,
        preferred_scales=BUSY_STREET_SCALES,
        gap_cm=0.0,
        setback_cm=setback_cm,
        facade_fill_ratio=facade_fill_ratio,
        height_factors=BUSY_STREET_HEIGHT_FACTORS,
        target_count=target_count,
    )
    return decorate_busy_street_placements(placements)


def plan_busy_street_props(
    buildings: tuple[BusyStreetBuilding, ...],
    start: tuple[float, float],
    end: tuple[float, float],
    *,
    outward: tuple[float, float],
    setback_cm: float = 200.0,
    frontage_offset_cm: float = 350.0,
) -> tuple[BusyStreetProp, ...]:
    """Place readable use cues without interrupting the walking corridor.

    Each restaurant receives one outdoor table. Every second house receives a
    small bin or hydrant, producing city-street detail without turning the
    frontage into a field of unrelated props.
    """

    tangent_raw = (float(end[0]) - float(start[0]), float(end[1]) - float(start[1]))
    tangent_length = math.hypot(*tangent_raw)
    outward_raw = (float(outward[0]), float(outward[1]))
    outward_length = math.hypot(*outward_raw)
    if tangent_length <= 1e-9 or outward_length <= 1e-9:
        raise ValueError("Street direction and outward vectors must be non-zero")
    tangent = (tangent_raw[0] / tangent_length, tangent_raw[1] / tangent_length)
    normal = (outward_raw[0] / outward_length, outward_raw[1] / outward_length)
    if abs(tangent[0] * normal[0] + tangent[1] * normal[1]) > 1e-6:
        raise ValueError("outward must be perpendicular to the street row")
    if not math.isfinite(setback_cm) or setback_cm < 0.0:
        raise ValueError("setback_cm must be finite and non-negative")
    if not math.isfinite(frontage_offset_cm) or frontage_offset_cm <= 0.0:
        raise ValueError("frontage_offset_cm must be finite and positive")

    props: list[BusyStreetProp] = []
    restaurant_count = 0
    house_count = 0
    for building in buildings:
        placement = building.placement
        boundary = (
            placement.position[0]
            + normal[0] * (placement.normal_half_extent_cm + setback_cm),
            placement.position[1]
            + normal[1] * (placement.normal_half_extent_cm + setback_cm),
        )
        if building.use == "restaurant":
            asset_key = (
                "BP_Table_C" if restaurant_count % 2 == 0 else "BP_Table2_C"
            )
            restaurant_count += 1
            use: StreetPropUse = "restaurant_seating"
            tangent_offset = placement.tangent_width_cm * (
                -0.20 if restaurant_count % 2 == 0 else 0.20
            )
        elif building.use == "bookshop":
            asset_key = "BP_Table2_C"
            use = "book_display"
            tangent_offset = placement.tangent_width_cm * 0.18
        elif building.use == "bar":
            asset_key = "BP_Table_C"
            use = "bar_seating"
            tangent_offset = -placement.tangent_width_cm * 0.18
        elif building.use == "skyscraper_lobby":
            continue
        else:
            house_count += 1
            if house_count % 2 != 0:
                continue
            asset_key = (
                "BP_Trash_bin_a_C" if house_count % 4 == 2 else "BP_Hydrant_C"
            )
            use = "street_furniture"
            tangent_offset = placement.tangent_width_cm * 0.28
        position = (
            boundary[0] + tangent[0] * tangent_offset + normal[0] * frontage_offset_cm,
            boundary[1] + tangent[1] * tangent_offset + normal[1] * frontage_offset_cm,
            0.0,
        )
        props.append(
            BusyStreetProp(
                index=len(props),
                building_index=placement.index,
                asset_key=asset_key,
                position=position,
                yaw_deg=placement.yaw_deg,
                scale=BUSY_STREET_PROP_SCALES[asset_key],
                use=use,
            )
        )
    return tuple(props)


__all__ = [
    "BAR_ASSETS",
    "BOOKSHOP_ASSETS",
    "BUSY_STREET_HEIGHT_FACTORS",
    "BUSY_STREET_MODULES",
    "BUSY_STREET_PROP_SCALES",
    "BUSY_STREET_SCALES",
    "BUSY_STREET_SEQUENCE",
    "BUSY_STREET_VENUE_MASKS",
    "BusyStreetBuilding",
    "BusyStreetProp",
    "FacadeUse",
    "HOUSE_ASSETS",
    "RESTAURANT_ASSETS",
    "SKYSCRAPER_ASSETS",
    "StreetFacadeModule",
    "StreetPropUse",
    "decorate_busy_street_placements",
    "plan_busy_street",
    "plan_busy_street_modules",
    "plan_busy_street_props",
]
