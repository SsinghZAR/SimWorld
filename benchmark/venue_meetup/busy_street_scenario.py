"""Shared scenario records for blocks built from busy-street facades."""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Iterable, Mapping
from typing import cast

from benchmark.venue_meetup.building_catalog import asset_path, building_description
from benchmark.venue_meetup.busy_street import BusyStreetBuilding, BusyStreetProp
from benchmark.venue_meetup.city_block import CityBlockPlan
from benchmark.venue_meetup.city_block_layout import (
    frontage_geometry,
    venue_slot_id,
)
from benchmark.venue_meetup.scenario import (
    Entrance,
    PropSpec,
    Region,
    Requirement,
    StaticBuilding,
    Venue,
    VenueProperties,
    VenueType,
)

PROP_MASK_COLORS = {
    "restaurant_seating": (255, 230, 40),
    "book_display": (50, 220, 255),
    "bar_seating": (255, 80, 80),
    "street_furniture": (80, 255, 120),
}


def index_props(
    props: Iterable[BusyStreetProp],
) -> dict[int, list[BusyStreetProp]]:
    """Group authored props by their stable facade index."""

    result: dict[int, list[BusyStreetProp]] = defaultdict(list)
    for prop in props:
        result[prop.building_index].append(prop)
    return dict(result)


def venue_properties(use: str, occurrence: int) -> VenueProperties:
    """Return deterministic hidden traits for one authored venue use."""

    if use == "restaurant":
        return VenueProperties(
            open=True,
            reachable=True,
            capacity=8 + occurrence,
            accessible=occurrence != 4,
            shelter=True,
            food_drink=True,
            quiet_score=max(0.25, 0.68 - 0.07 * occurrence),
            crowding_score=min(0.85, 0.32 + 0.08 * occurrence),
            near_transit=False,
        )
    if use == "bookshop":
        return VenueProperties(
            open=True,
            reachable=True,
            capacity=6,
            accessible=True,
            shelter=True,
            food_drink=False,
            quiet_score=0.88 - 0.05 * occurrence,
            crowding_score=0.22 + 0.08 * occurrence,
            near_transit=False,
        )
    if use == "bar":
        return VenueProperties(
            open=True,
            reachable=True,
            capacity=12,
            accessible=occurrence == 0,
            shelter=True,
            food_drink=True,
            quiet_score=0.24,
            crowding_score=0.78,
            near_transit=False,
        )
    if use == "skyscraper_lobby":
        return VenueProperties(
            open=True,
            reachable=True,
            capacity=18,
            accessible=True,
            shelter=True,
            food_drink=occurrence == 1,
            quiet_score=0.62,
            crowding_score=0.48,
            near_transit=True,
        )
    raise ValueError(f"Unsupported playtest venue use: {use}")


def playtest_requirements() -> list[Requirement]:
    """Return the shared social requirements for facade playtest scenarios."""

    return [
        Requirement("open", 2.0, hard=True, description="Venue must be open."),
        Requirement(
            "reachable", 2.0, hard=True, description="Venue must be reachable."
        ),
        Requirement(
            "accessible",
            2.0,
            hard=True,
            description="Step-free access is required.",
        ),
        Requirement("food_drink", 1.25, description="Food or drink is useful."),
        Requirement("quiet", 1.0, description="Quieter venues are preferred."),
        Requirement("shelter", 0.75, description="Shelter is preferred."),
    ]


def static_residence(
    building: BusyStreetBuilding,
    *,
    namespace: str = "",
) -> StaticBuilding:
    """Convert one non-interactive facade to a solid scenario building."""

    placement = building.placement
    prefix = f"{namespace}_" if namespace else ""
    return StaticBuilding(
        building_id=f"{prefix}residence_{placement.index:02d}",
        asset_key=placement.asset_key,
        asset_path=asset_path(placement.asset_key),
        position=(*placement.position, 0.0),
        yaw_deg=placement.yaw_deg,
        scale=placement.scale,
        visual_summary=building_description(placement.asset_key),
    )


def interactive_venue(
    plan: CityBlockPlan,
    building: BusyStreetBuilding,
    *,
    occurrence: int,
    props_by_building: Mapping[int, list[BusyStreetProp]],
    zone_id: str,
) -> Venue:
    """Convert one commercial facade to a complete inspectable venue."""

    placement = building.placement
    properties = venue_properties(building.use, occurrence)
    slot_id = venue_slot_id(building)
    display_name = building.display_name or slot_id.replace("_", " ").title()
    geometry = frontage_geometry(plan, building)
    if building.venue_id is None or building.mask_color_rgb is None:
        raise ValueError("Interactive facade requires venue identity and mask color")
    props = [
        PropSpec(
            prop_id=f"{building.venue_id}_{prop.use}_{local_index}",
            asset_key=prop.asset_key,
            position=prop.position,
            yaw_deg=prop.yaw_deg,
            scale=prop.scale,
            semantic=prop.use,
            color_rgb=PROP_MASK_COLORS[prop.use],
        )
        for local_index, prop in enumerate(
            props_by_building.get(building.placement.index, [])
        )
    ]
    return Venue(
        venue_id=building.venue_id,
        slot_id=slot_id,
        venue_type=cast(VenueType, building.use),
        asset_key=placement.asset_key,
        asset_path=asset_path(placement.asset_key),
        position=(*placement.position, 0.0),
        yaw_deg=placement.yaw_deg,
        region=Region(center=geometry.meeting, radius=350.0),
        mask_color_rgb=building.mask_color_rgb,
        properties=properties,
        entrances=[
            Entrance(
                entrance_id=f"{slot_id}_front_door",
                status="accessible" if properties.accessible else "stairs_only",
                position=(*geometry.boundary, 0.0),
                yaw_deg=placement.yaw_deg,
                visible_cues=[display_name, building.visual_cue],
            )
        ],
        props=props,
        visual_summary=(
            f"{display_name}: {building.visual_cue}. "
            f"{building_description(placement.asset_key)}"
        ),
        scale=placement.scale,
        zone_id=zone_id,
    )


__all__ = [
    "PROP_MASK_COLORS",
    "index_props",
    "interactive_venue",
    "playtest_requirements",
    "static_residence",
    "venue_properties",
]
