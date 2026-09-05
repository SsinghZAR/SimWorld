"""Rosebank-inspired 9x9 mixed-use navigation district."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.venue_meetup.building_catalog import (
    assert_catalog_assets_exist,
    asset_path,
    building_description,
)
from benchmark.venue_meetup.busy_street import BUSY_STREET_VENUE_MASKS
from benchmark.venue_meetup.busy_street_scenario import playtest_requirements
from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.rosebank_grid import (
    LANDMARK_BLOCK_ROLES,
    RosebankGridPlan,
    RosebankVenueSite,
    plan_rosebank_grid,
)
from benchmark.venue_meetup.rosebank_grid_layout import (
    build_rosebank_grid_layout,
    intersection_node_id,
)
from benchmark.venue_meetup.rosebank_grid_massing import plan_rosebank_massing
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Entrance,
    Landmark,
    LandmarkType,
    Region,
    Scenario,
    Venue,
    VenueProperties,
)

MAP_TEMPLATE_ID = "rosebank_grid_9x9_v0"
LAYOUT_ID = MAP_TEMPLATE_ID
AGENT_Z = 150.0


@dataclass(frozen=True, slots=True)
class _LandmarkDefinition:
    display_name: str
    landmark_type: LandmarkType
    asset_key: str
    scale: tuple[float, float, float]


_LANDMARKS = {
    "clock_tower": _LandmarkDefinition(
        "West Rosebank Clock Tower",
        "clock_tower",
        "BP_Building_20_C",
        (0.62, 0.62, 0.92),
    ),
    "arts_centre": _LandmarkDefinition(
        "Keyes Arts Centre",
        "museum",
        "BP_Building_99_C",
        (0.38, 0.38, 0.62),
    ),
    "market_hall": _LandmarkDefinition(
        "Rosebank Market Hall",
        "venue_hall",
        "BP_Building_123_C",
        (0.28, 0.28, 0.52),
    ),
    "gautrain_tower": _LandmarkDefinition(
        "Oxford Gautrain Tower",
        "commercial_tower",
        "BP_Building_101_C",
        (0.46, 0.46, 1.12),
    ),
    "hotel_tower": _LandmarkDefinition(
        "Tyrwhitt Hotel Tower",
        "hotel",
        "BP_Building_95_C",
        (0.46, 0.46, 0.78),
    ),
    "civic_hall": _LandmarkDefinition(
        "East Rosebank Civic Hall",
        "hospital",
        "BP_Building_87_C",
        (0.42, 0.42, 0.56),
    ),
}


def plan_playtest_grid() -> RosebankGridPlan:
    """Return the canonical Rosebank-inspired 81-block plan."""

    return plan_rosebank_grid()


def build_district_layout(
    plan: RosebankGridPlan | None = None,
) -> DistrictLayout:
    """Return the public street, block, frontage, and alley graph."""

    return build_rosebank_grid_layout(
        plan or plan_playtest_grid(),
        layout_id=LAYOUT_ID,
    )


def _venue_properties(
    plan: RosebankGridPlan,
    site: RosebankVenueSite,
    index: int,
) -> VenueProperties:
    block = plan.block_by_id(site.block_id)
    food_drink = site.venue_type in {
        "restaurant",
        "cafe",
        "bar",
        "pub",
        "hotel_lobby",
    }
    if index == 0:
        return VenueProperties(
            open=True,
            reachable=True,
            capacity=14,
            accessible=True,
            shelter=True,
            food_drink=True,
            quiet_score=0.78,
            crowding_score=0.30,
            near_transit=True,
        )
    base_quiet = {
        "residential": 0.82,
        "mixed_use": 0.61,
        "transit_core": 0.40,
        "civic": 0.66,
        "garden": 0.88,
    }[block.zone]
    variation = ((index % 5) - 2) * 0.045
    quiet_score = max(0.18, min(0.92, base_quiet + variation))
    crowding_score = max(
        0.12,
        min(0.88, 0.94 - quiet_score + (index % 3) * 0.04),
    )
    return VenueProperties(
        open=index % 11 != 10,
        reachable=True,
        capacity=6 + (index % 9) * 2,
        accessible=index % 6 != 5,
        shelter=index % 8 != 7,
        food_drink=food_drink,
        quiet_score=quiet_score,
        crowding_score=crowding_score,
        near_transit=max(abs(block.row - 4), abs(block.column - 4)) <= 2,
    )


def _build_landmarks(plan: RosebankGridPlan) -> list[Landmark]:
    landmarks: list[Landmark] = []
    for index, (block_id, role) in enumerate(LANDMARK_BLOCK_ROLES.items()):
        block = plan.block_by_id(block_id)
        definition = _LANDMARKS[role]
        landmarks.append(
            Landmark(
                landmark_id=f"landmark_{role}",
                slot_id=role,
                landmark_type=definition.landmark_type,
                asset_key=definition.asset_key,
                asset_path=asset_path(definition.asset_key),
                position=(*block.center, 0.0),
                yaw_deg=float((index * 90) % 360),
                mask_color_rgb=BUSY_STREET_VENUE_MASKS[36 + index],
                visual_summary=(
                    f"{definition.display_name}. "
                    f"{building_description(definition.asset_key)}"
                ),
                scale=definition.scale,
            )
        )
    return landmarks


def build_fixed_scenario(seed: int = 17) -> Scenario:
    """Return the deterministic 9x9 mixed-use navigation scenario."""

    plan = plan_playtest_grid()
    layout = build_district_layout(plan)
    frontage_by_slot = {
        frontage.venue_slot_id: frontage
        for frontage in layout.frontages
        if frontage.venue_slot_id is not None
    }
    venues: list[Venue] = []
    for index, site in enumerate(plan.venue_sites):
        frontage = frontage_by_slot[site.slot_id]
        properties = _venue_properties(plan, site, index)
        venues.append(
            Venue(
                venue_id=site.venue_id,
                slot_id=site.slot_id,
                venue_type=site.venue_type,
                asset_key=site.asset_key,
                asset_path=asset_path(site.asset_key),
                position=frontage.position,
                yaw_deg=frontage.yaw_deg,
                region=Region(
                    center=frontage.meeting_region.center,
                    radius=frontage.meeting_region.radius,
                ),
                mask_color_rgb=BUSY_STREET_VENUE_MASKS[index],
                properties=properties,
                entrances=[
                    Entrance(
                        entrance_id=f"{site.slot_id}_front_door",
                        status=(
                            "accessible"
                            if properties.accessible
                            else "stairs_only"
                        ),
                        position=frontage.entrance_point,
                        yaw_deg=frontage.yaw_deg,
                        visible_cues=[
                            site.display_name,
                            f"Block {site.block_id}",
                            plan.block_by_id(site.block_id).zone.replace("_", " "),
                        ],
                    )
                ],
                props=[],
                visual_summary=(
                    f"{site.display_name}: "
                    f"{building_description(site.asset_key)}"
                ),
                scale=site.scale,
                zone_id=site.zone_id,
            )
        )

    landmarks = _build_landmarks(plan)
    buildings = list(plan_rosebank_massing(plan))
    assert_catalog_assets_exist(
        [
            *(venue.asset_key for venue in venues),
            *(landmark.asset_key for landmark in landmarks),
            *(building.asset_key for building in buildings),
        ]
    )
    west_spawn_id = intersection_node_id(0, 5)
    east_spawn_id = intersection_node_id(9, 5)
    west_spawn = layout.node_by_id(west_spawn_id).position
    east_spawn = layout.node_by_id(east_spawn_id).position
    agents = [
        AgentSpec(
            agent_id="agent_0",
            spawn_slot="west_gateway",
            position=(*west_spawn, AGENT_Z),
            yaw_deg=0.0,
            private_constraint="I need step-free access and cannot use stairs.",
            private_requirement_keys=["accessible"],
            zone_id="zone_west",
            walk_node_id=west_spawn_id,
        ),
        AgentSpec(
            agent_id="agent_1",
            spawn_slot="east_gateway",
            position=(*east_spawn, AGENT_Z),
            yaw_deg=180.0,
            private_constraint="I strongly prefer food or drink and a quiet place.",
            private_requirement_keys=["food_drink", "quiet"],
            zone_id="zone_east",
            walk_node_id=east_spawn_id,
        ),
    ]
    return Scenario(
        scenario_id=f"rosebank_grid_9x9_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        seed=seed,
        venues=venues,
        landmarks=landmarks,
        agents=agents,
        requirements=playtest_requirements(),
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text=(
            "Rosebank-inspired 9x9 grid: Oxford Road is the north-south transit "
            "spine; Tyrwhitt is the east-west high street. Dense office and "
            "retail blocks cluster around the central station, mixed-use blocks "
            "step down to leafy residential edges, and brown mid-block alleys "
            "provide recognizable shortcuts. Landmark towers, the market hall, "
            "arts centre, civic hall, hotel, and four garden blocks support "
            "relative positioning. Venue suitability remains hidden until "
            "inspection."
        ),
        max_steps=384,
        layout=layout,
        buildings=buildings,
    )


__all__ = [
    "LAYOUT_ID",
    "MAP_TEMPLATE_ID",
    "build_district_layout",
    "build_fixed_scenario",
    "plan_playtest_grid",
]
