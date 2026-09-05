"""Three dense city blocks connected through aligned pedestrian alleys."""

from __future__ import annotations

from collections import defaultdict

from benchmark.venue_meetup.building_catalog import assert_catalog_assets_exist
from benchmark.venue_meetup.busy_street_scenario import (
    index_props,
    interactive_venue,
    playtest_requirements,
    static_residence,
)
from benchmark.venue_meetup.city_block_layout import OUTER_WALK_OFFSET_CM
from benchmark.venue_meetup.connected_blocks import (
    ConnectedBlocksPlan,
    plan_connected_block_props,
    plan_connected_blocks,
)
from benchmark.venue_meetup.connected_blocks_layout import (
    block_node_id,
    build_connected_blocks_layout,
)
from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Scenario,
    StaticBuilding,
    Venue,
)

MAP_TEMPLATE_ID = "connected_blocks_playtest_v0"
LAYOUT_ID = MAP_TEMPLATE_ID
AGENT_Z = 150.0


def plan_playtest_district() -> ConnectedBlocksPlan:
    """Return the canonical west–central–east three-block district."""

    return plan_connected_blocks()


def build_district_layout(
    district_plan: ConnectedBlocksPlan | None = None,
) -> DistrictLayout:
    """Return the namespaced block graphs plus their two alley links."""

    return build_connected_blocks_layout(
        district_plan or plan_playtest_district(),
        layout_id=LAYOUT_ID,
    )


def build_fixed_scenario(seed: int = 17) -> Scenario:
    """Return the deterministic multi-block alley navigation scenario."""

    plan = plan_playtest_district()
    assert_catalog_assets_exist(
        tuple(building.placement.asset_key for building in plan.buildings)
    )
    props_by_building = index_props(plan_connected_block_props(plan))
    occurrences: dict[str, int] = defaultdict(int)
    venues: list[Venue] = []
    static_buildings: list[StaticBuilding] = []
    for block in plan.blocks:
        for building in block.plan.buildings:
            if building.venue_id is None:
                static_buildings.append(
                    static_residence(building, namespace=block.block_id)
                )
                continue
            occurrence = occurrences[building.use]
            occurrences[building.use] += 1
            venues.append(
                interactive_venue(
                    block.plan,
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

    west = plan.block_by_id("west")
    east = plan.block_by_id("east")
    west_spawn = west.plan.portal_by_side("west").offset_position(
        OUTER_WALK_OFFSET_CM
    )
    east_spawn = east.plan.portal_by_side("east").offset_position(
        OUTER_WALK_OFFSET_CM
    )
    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="west_district_gate",
            position=(*west_spawn, AGENT_Z),
            yaw_deg=0.0,
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
            zone_id="zone_west",
            walk_node_id=block_node_id("west", "portal_west_outer"),
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="east_district_gate",
            position=(*east_spawn, AGENT_Z),
            yaw_deg=180.0,
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
            zone_id="zone_east",
            walk_node_id=block_node_id("east", "portal_east_outer"),
        ),
    ]

    return Scenario(
        scenario_id=f"connected_blocks_playtest_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        seed=seed,
        venues=venues,
        landmarks=[],
        agents=agents,
        requirements=playtest_requirements(),
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text=(
            "Coarse map: West Market, Central Arcade, and East Tower blocks "
            "form one dense east-west district. Every block has a perimeter "
            "sidewalk and internal courtyard. The west-central and central-east "
            "portal pairs are joined by narrow public alleys, so cross-district "
            "routes pass through multiple street walls rather than open terrain. "
            "Venue status and suitability remain hidden until inspection."
        ),
        max_steps=192,
        layout=build_district_layout(plan),
        buildings=static_buildings,
    )


__all__ = [
    "LAYOUT_ID",
    "MAP_TEMPLATE_ID",
    "build_district_layout",
    "build_fixed_scenario",
    "plan_playtest_district",
]
