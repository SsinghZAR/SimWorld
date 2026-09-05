"""Scalable Rosebank-inspired mixed-use navigation districts."""

from __future__ import annotations

from dataclasses import dataclass

from benchmark.venue_meetup.building_catalog import (
    assert_catalog_assets_exist, asset_path, building_description)
from benchmark.venue_meetup.busy_street import BUSY_STREET_VENUE_MASKS
from benchmark.venue_meetup.busy_street_scenario import playtest_requirements
from benchmark.venue_meetup.layout import DistrictLayout
from benchmark.venue_meetup.rosebank_grid import (ROSEBANK_GRID_MAX_STEPS,
                                                  ROSEBANK_GRID_TEMPLATE_IDS,
                                                  SUPPORTED_GRID_SIZES,
                                                  RosebankGridPlan,
                                                  RosebankVenueSite,
                                                  plan_rosebank_grid)
from benchmark.venue_meetup.rosebank_grid_layout import (
    build_rosebank_grid_layout, intersection_node_id)
from benchmark.venue_meetup.rosebank_grid_massing import plan_rosebank_massing
from benchmark.venue_meetup.scenario import (AgentSpec, Entrance, Landmark,
                                             LandmarkType, Region, Scenario,
                                             Venue, VenueProperties)

MAP_TEMPLATE_ID = ROSEBANK_GRID_TEMPLATE_IDS[9]
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


def plan_playtest_grid(grid_size: int = 9) -> RosebankGridPlan:
    """Return a canonical Rosebank-inspired plan at a supported scale."""

    return plan_rosebank_grid(grid_size=grid_size)


def build_district_layout(
    plan: RosebankGridPlan | None = None,
    *,
    layout_id: str | None = None,
) -> DistrictLayout:
    """Return the public street, block, frontage, and alley graph."""

    resolved_plan = plan or plan_playtest_grid()
    return build_rosebank_grid_layout(
        resolved_plan,
        layout_id=layout_id or ROSEBANK_GRID_TEMPLATE_IDS[resolved_plan.grid_size],
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
        near_transit=max(
            abs(block.row - plan.grid_size // 2),
            abs(block.column - plan.grid_size // 2),
        ) <= max(1, plan.grid_size // 4),
    )


def _build_landmarks(plan: RosebankGridPlan) -> list[Landmark]:
    landmarks: list[Landmark] = []
    landmark_blocks = tuple(
        block for block in plan.blocks if block.landmark_role is not None
    )
    for index, block in enumerate(landmark_blocks):
        role = block.landmark_role
        assert role is not None
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
                mask_color_rgb=BUSY_STREET_VENUE_MASKS[
                    len(plan.venue_sites) + index
                ],
                visual_summary=(
                    f"{definition.display_name}. "
                    f"{building_description(definition.asset_key)}"
                ),
                scale=definition.scale,
            )
        )
    return landmarks


def _coarse_map_text(plan: RosebankGridPlan) -> str:
    """Describe only public structure that actually exists at this scale."""

    garden_count = sum(block.zone == "garden" for block in plan.blocks)
    residential = any(block.zone == "residential" for block in plan.blocks)
    district_form = (
        "Dense office and retail blocks cluster around the central station, "
        "then step down to quieter residential edges."
        if residential
        else "Compact office, retail, and civic blocks cluster around the "
        "central station."
    )
    garden_note = (
        f" {garden_count} garden blocks provide additional green anchors."
        if garden_count
        else ""
    )
    return (
        f"Rosebank-inspired {plan.grid_size}x{plan.grid_size} grid with "
        f"{len(plan.venue_sites)} candidate venues: Oxford Road is the "
        "north-south transit spine; Tyrwhitt is the east-west high street. "
        f"{district_form} Brown mid-block alleys provide recognizable "
        "shortcuts. Distinctive landmark buildings support relative "
        f"positioning.{garden_note} Venue suitability remains hidden until "
        "inspection."
    )


def build_scaled_scenario(*, grid_size: int, seed: int = 17) -> Scenario:
    """Return one deterministic Rosebank grid tier as a complete scenario."""

    if grid_size not in SUPPORTED_GRID_SIZES:
        raise ValueError(
            f"Unsupported Rosebank grid size {grid_size}; expected one of "
            f"{SUPPORTED_GRID_SIZES}"
        )
    template_id = ROSEBANK_GRID_TEMPLATE_IDS[grid_size]
    plan = plan_playtest_grid(grid_size)
    layout = build_district_layout(plan, layout_id=template_id)
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
    west_spawn_id = intersection_node_id(0, plan.primary_street_index)
    east_spawn_id = intersection_node_id(
        plan.grid_size,
        plan.primary_street_index,
    )
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
        scenario_id=f"rosebank_grid_{grid_size}x{grid_size}_seed_{seed}",
        map_template_id=template_id,
        seed=seed,
        venues=venues,
        landmarks=landmarks,
        agents=agents,
        requirements=playtest_requirements(),
        soft_weights={"quiet_threshold": 0.65, "crowding_threshold": 0.5},
        coarse_map_text=_coarse_map_text(plan),
        max_steps=ROSEBANK_GRID_MAX_STEPS[grid_size],
        layout=layout,
        buildings=buildings,
    )


def build_3x3_scenario(seed: int = 17) -> Scenario:
    """Return the compact 3x3 / four-venue benchmark tier."""

    return build_scaled_scenario(grid_size=3, seed=seed)


def build_5x5_scenario(seed: int = 17) -> Scenario:
    """Return the intermediate 5x5 / eight-venue benchmark tier."""

    return build_scaled_scenario(grid_size=5, seed=seed)


def build_7x7_scenario(seed: int = 17) -> Scenario:
    """Return the large 7x7 / twelve-venue benchmark tier."""

    return build_scaled_scenario(grid_size=7, seed=seed)


def build_fixed_scenario(seed: int = 17) -> Scenario:
    """Return the original 9x9 / 36-venue playtest for compatibility."""

    return build_scaled_scenario(grid_size=9, seed=seed)


__all__ = [
    "LAYOUT_ID",
    "MAP_TEMPLATE_ID",
    "build_3x3_scenario",
    "build_5x5_scenario",
    "build_7x7_scenario",
    "build_district_layout",
    "build_fixed_scenario",
    "build_scaled_scenario",
    "plan_playtest_grid",
]
