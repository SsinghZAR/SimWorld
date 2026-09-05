"""Interactive four-entry city block used for visual and navigation playtests.

All 24 authored facades form one compact perimeter. Twelve commercial facades
remain Venue records and twelve residences remain solid StaticBuilding
obstacles. Centred openings connect the outer sidewalk to a courtyard ring.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.venue_meetup.building_catalog import (
    assert_catalog_assets_exist,
    asset_path,
    building_description,
)
from benchmark.venue_meetup.busy_street import BusyStreetBuilding, BusyStreetProp
from benchmark.venue_meetup.city_block import (
    CityBlockPlan,
    DEFAULT_BLOCK_SIDE_LENGTH_CM,
    DEFAULT_FACADE_FILL_RATIO,
    DEFAULT_PORTAL_WIDTH_CM,
    DEFAULT_SETBACK_CM,
    plan_city_block,
    plan_city_block_props,
)
from benchmark.venue_meetup.city_block_layout import (
    COURTYARD_RING_OFFSET_CM,
    OUTER_WALK_OFFSET_CM,
    build_city_block_layout,
    frontage_geometry,
    venue_slot_id,
)
from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Entrance,
    PropSpec,
    Region,
    Requirement,
    Scenario,
    StaticBuilding,
    Venue,
    VenueProperties,
)

MAP_TEMPLATE_ID = "busy_street_playtest_v0"
LAYOUT_ID = MAP_TEMPLATE_ID
BLOCK_SIDE_LENGTH_CM = DEFAULT_BLOCK_SIDE_LENGTH_CM
PORTAL_WIDTH_CM = DEFAULT_PORTAL_WIDTH_CM
FACADE_FILL_RATIO = DEFAULT_FACADE_FILL_RATIO
SETBACK_CM = DEFAULT_SETBACK_CM
BLOCK_HALF_EXTENT_CM = BLOCK_SIDE_LENGTH_CM / 2.0
AGENT_Z = 150.0

# Compatibility aliases for callers of the earlier straight north-wall API.
WALL_LENGTH_CM = BLOCK_SIDE_LENGTH_CM
WALL_Y_CM = BLOCK_HALF_EXTENT_CM
WALL_START = (-BLOCK_HALF_EXTENT_CM, BLOCK_HALF_EXTENT_CM)
WALL_END = (BLOCK_HALF_EXTENT_CM, BLOCK_HALF_EXTENT_CM)
OUTWARD = (0.0, 1.0)

_PROP_MASK_COLORS = {
    "restaurant_seating": (255, 230, 40),
    "book_display": (50, 220, 255),
    "bar_seating": (255, 80, 80),
    "street_furniture": (80, 255, 120),
}


def plan_playtest_block() -> CityBlockPlan:
    """Return the canonical 24-facade, four-portal playtest block."""

    return plan_city_block(
        side_length_cm=BLOCK_SIDE_LENGTH_CM,
        portal_width_cm=PORTAL_WIDTH_CM,
        facade_fill_ratio=FACADE_FILL_RATIO,
        setback_cm=SETBACK_CM,
    )


def plan_playtest_street() -> tuple[BusyStreetBuilding, ...]:
    """Compatibility view of every facade in the four-sided block."""

    return plan_playtest_block().buildings


def build_district_layout(
    block_plan: CityBlockPlan | None = None,
) -> DistrictLayout:
    """Return the authored perimeter, portal, and courtyard walk graph."""

    return build_city_block_layout(
        block_plan or plan_playtest_block(),
        layout_id=LAYOUT_ID,
    )


def _properties(use: str, occurrence: int) -> VenueProperties:
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


def _venue_props(
    building: BusyStreetBuilding,
    props_by_building: dict[int, list[BusyStreetProp]],
) -> list[PropSpec]:
    if building.venue_id is None:
        raise ValueError("Residential facade cannot own venue props")
    return [
        PropSpec(
            prop_id=f"{building.venue_id}_{prop.use}_{local_index}",
            asset_key=prop.asset_key,
            position=prop.position,
            yaw_deg=prop.yaw_deg,
            scale=prop.scale,
            semantic=prop.use,
            color_rgb=_PROP_MASK_COLORS[prop.use],
        )
        for local_index, prop in enumerate(
            props_by_building.get(building.placement.index, [])
        )
    ]


def _static_residence(building: BusyStreetBuilding) -> StaticBuilding:
    placement = building.placement
    return StaticBuilding(
        building_id=f"residence_{placement.index:02d}",
        asset_key=placement.asset_key,
        asset_path=asset_path(placement.asset_key),
        position=(*placement.position, 0.0),
        yaw_deg=placement.yaw_deg,
        scale=placement.scale,
        visual_summary=building_description(placement.asset_key),
    )


def _interactive_venue(
    plan: CityBlockPlan,
    building: BusyStreetBuilding,
    *,
    occurrence: int,
    props_by_building: dict[int, list[BusyStreetProp]],
) -> Venue:
    placement = building.placement
    properties = _properties(building.use, occurrence)
    slot_id = venue_slot_id(building)
    display_name = building.display_name or slot_id.replace("_", " ").title()
    geometry = frontage_geometry(plan, building)
    if building.venue_id is None or building.mask_color_rgb is None:
        raise ValueError("Interactive facade requires venue identity and mask color")
    return Venue(
        venue_id=building.venue_id,
        slot_id=slot_id,
        venue_type=building.use,  # type: ignore[arg-type]
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
        props=_venue_props(building, props_by_building),
        visual_summary=(
            f"{display_name}: {building.visual_cue}. "
            f"{building_description(placement.asset_key)}"
        ),
        scale=placement.scale,
        zone_id="zone_west" if placement.position[0] < 0.0 else "zone_east",
    )


def build_fixed_scenario(seed: int = 17) -> Scenario:
    """Return the deterministic interactive city-block playtest scenario."""

    plan = plan_playtest_block()
    assert_catalog_assets_exist(
        tuple(building.placement.asset_key for building in plan.buildings)
    )
    props_by_building: dict[int, list[BusyStreetProp]] = defaultdict(list)
    for prop in plan_city_block_props(plan):
        props_by_building[prop.building_index].append(prop)

    occurrences: dict[str, int] = defaultdict(int)
    venues: list[Venue] = []
    static_buildings: list[StaticBuilding] = []
    for building in plan.buildings:
        if building.venue_id is None:
            static_buildings.append(_static_residence(building))
            continue
        occurrence = occurrences[building.use]
        occurrences[building.use] += 1
        venues.append(
            _interactive_venue(
                plan,
                building,
                occurrence=occurrence,
                props_by_building=props_by_building,
            )
        )

    west_spawn = plan.portal_by_side("west").offset_position(
        OUTER_WALK_OFFSET_CM
    )
    east_spawn = plan.portal_by_side("east").offset_position(
        OUTER_WALK_OFFSET_CM
    )
    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="west_gate",
            position=(*west_spawn, AGENT_Z),
            yaw_deg=0.0,
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
            zone_id="zone_west",
            walk_node_id="portal_west_outer",
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="east_gate",
            position=(*east_spawn, AGENT_Z),
            yaw_deg=180.0,
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
            zone_id="zone_east",
            walk_node_id="portal_east_outer",
        ),
    ]

    return Scenario(
        scenario_id=f"busy_street_playtest_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        seed=seed,
        venues=venues,
        landmarks=[],
        agents=agents,
        requirements=[
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
        ],
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text=(
            "Coarse map: one compact mixed-use city block enclosed by 24 "
            "continuous facades. Centred north, east, south, and west portals "
            "connect the perimeter sidewalk to an internal courtyard ring. "
            "Twelve named venues—six restaurants, two bookshops, two bars, "
            "and two skyscraper lobbies—alternate with solid residences. "
            "Venue status and suitability remain hidden until inspection."
        ),
        max_steps=96,
        layout=build_district_layout(plan),
        buildings=static_buildings,
    )


__all__ = [
    "BLOCK_HALF_EXTENT_CM",
    "BLOCK_SIDE_LENGTH_CM",
    "COURTYARD_RING_OFFSET_CM",
    "LAYOUT_ID",
    "MAP_TEMPLATE_ID",
    "OUTER_WALK_OFFSET_CM",
    "PORTAL_WIDTH_CM",
    "WALL_END",
    "WALL_LENGTH_CM",
    "WALL_START",
    "build_district_layout",
    "build_fixed_scenario",
    "plan_playtest_block",
    "plan_playtest_street",
]
