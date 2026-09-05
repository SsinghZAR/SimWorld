"""Interactive four-entry city block used for visual and navigation playtests.

All 24 authored facades form one compact perimeter. Twelve commercial facades
remain Venue records and twelve residences remain solid StaticBuilding
obstacles. Centred openings connect the outer sidewalk to a courtyard ring.
"""

from __future__ import annotations

from collections import defaultdict

from benchmark.venue_meetup.building_catalog import assert_catalog_assets_exist
from benchmark.venue_meetup.busy_street import BusyStreetBuilding
from benchmark.venue_meetup.busy_street_scenario import (
    index_props,
    interactive_venue,
    playtest_requirements,
    static_residence,
)
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
)
from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Scenario,
    StaticBuilding,
    Venue,
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


def build_fixed_scenario(seed: int = 17) -> Scenario:
    """Return the deterministic interactive city-block playtest scenario."""

    plan = plan_playtest_block()
    assert_catalog_assets_exist(
        tuple(building.placement.asset_key for building in plan.buildings)
    )
    props_by_building = index_props(plan_city_block_props(plan))
    occurrences: dict[str, int] = defaultdict(int)
    venues: list[Venue] = []
    static_buildings: list[StaticBuilding] = []
    for building in plan.buildings:
        if building.venue_id is None:
            static_buildings.append(static_residence(building))
            continue
        occurrence = occurrences[building.use]
        occurrences[building.use] += 1
        venues.append(
            interactive_venue(
                plan,
                building,
                occurrence=occurrence,
                props_by_building=props_by_building,
                zone_id=(
                    "zone_west"
                    if building.placement.position[0] < 0.0
                    else "zone_east"
                ),
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
        requirements=playtest_requirements(),
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
