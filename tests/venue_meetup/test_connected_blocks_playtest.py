"""Contracts for the three-block alley navigation playtest template."""

from __future__ import annotations

from collections import Counter
from itertools import combinations
from math import dist

import pytest

from benchmark.venue_meetup.connected_blocks import (
    DEFAULT_BLOCK_GAP_CM,
    plan_connected_block_props,
)
from benchmark.venue_meetup.connected_blocks_layout import block_node_id
from benchmark.venue_meetup.district_dressing import plan_district_actors
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.navigation import (
    building_obstacles,
    meeting_target,
    plan_layout_route,
)
from benchmark.venue_meetup.template_validation import collect_layout_errors
from benchmark.venue_meetup.templates.connected_blocks_playtest import (
    MAP_TEMPLATE_ID,
    build_fixed_scenario,
    plan_playtest_district,
)
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
def test_three_blocks_have_unique_facades_and_two_clear_alleys() -> None:
    plan = plan_playtest_district()

    assert [block.block_id for block in plan.blocks] == [
        "west",
        "central",
        "east",
    ]
    assert plan.block_gap_cm == DEFAULT_BLOCK_GAP_CM
    assert [block.plan.module_cycle for block in plan.blocks] == [0, 1, 2]
    assert len(plan.buildings) == 72
    assert sorted(
        building.placement.index for building in plan.buildings
    ) == list(range(72))
    north_side_assets = {
        tuple(
            building.placement.asset_key
            for building in block.plan.buildings_on_side("north")
        )
        for block in plan.blocks
    }
    assert len(north_side_assets) == 3
    for block in plan.blocks:
        assert len(block.plan.buildings) == 24
        assert len(block.plan.portals) == 4
        for side in ("north", "east", "south", "west"):
            assert len(block.plan.buildings_on_side(side)) == 6

    assert [alley.alley_id for alley in plan.alleys] == [
        "west_central_alley",
        "central_east_alley",
    ]
    for alley in plan.alleys:
        assert alley.length_cm == pytest.approx(700.0)
        assert alley.clear_width_cm >= 800.0


def test_connected_scenario_has_distinct_interactable_mix() -> None:
    scenario = build_fixed_scenario()

    assert scenario.map_template_id == MAP_TEMPLATE_ID
    assert len(scenario.venues) == 36
    assert len(scenario.buildings) == 36
    assert Counter(venue.venue_type for venue in scenario.venues) == {
        "restaurant": 18,
        "bookshop": 6,
        "bar": 6,
        "skyscraper_lobby": 6,
    }
    assert len({venue.venue_id for venue in scenario.venues}) == 36
    assert len({venue.slot_id for venue in scenario.venues}) == 36
    assert len({venue.mask_color_rgb for venue in scenario.venues}) == 36
    assert len(
        {prop.prop_id for venue in scenario.venues for prop in venue.props}
    ) == sum(len(venue.props) for venue in scenario.venues)
    assert Counter(venue.zone_id for venue in scenario.venues) == {
        "zone_west": 18,
        "zone_east": 18,
    }

    for left, right in combinations(
        (venue.mask_color_rgb for venue in scenario.venues), 2
    ):
        assert max(abs(left[index] - right[index]) for index in range(3)) > 8
        assert max(abs(left[index] - right[2 - index]) for index in range(3)) > 8


def test_all_facade_props_have_global_report_indices() -> None:
    props = plan_connected_block_props(plan_playtest_district())

    assert [prop.index for prop in props] == list(range(len(props)))
    assert len({prop.building_index for prop in props}) == len(props)


def test_layout_reaches_every_venue_and_crosses_both_alleys() -> None:
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    required_paths: list[tuple[str, str]] = []
    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        for frontage in scenario.layout.frontages:
            assert frontage.approach_node_id is not None
            required_paths.append(
                (agent.walk_node_id, frontage.approach_node_id)
            )
    assert collect_layout_errors(
        scenario.layout,
        required_paths=required_paths,
    ) == []
    assert plan_district_actors(scenario) == ()

    east_target = next(
        venue
        for venue in scenario.venues
        if venue.venue_id == "venue_red_awning_grill_3"
    )
    route = plan_layout_route(
        scenario.layout,
        scenario.agents[0].walk_node_id or "",
        venue_slot_id=east_target.slot_id,
    )
    assert route is not None
    assert block_node_id("west", "portal_east_outer") in route.node_ids
    assert block_node_id("central", "portal_west_outer") in route.node_ids
    assert block_node_id("central", "portal_east_outer") in route.node_ids
    assert block_node_id("east", "portal_west_outer") in route.node_ids

    connector_pairs = {
        frozenset((edge.start_node_id, edge.end_node_id)): edge
        for edge in scenario.layout.walk_edges
    }
    for left, right in (
        (
            block_node_id("west", "portal_east_outer"),
            block_node_id("central", "portal_west_outer"),
        ),
        (
            block_node_id("central", "portal_east_outer"),
            block_node_id("east", "portal_west_outer"),
        ),
    ):
        edge = connector_pairs[frozenset((left, right))]
        assert edge.route_kind == "alley"
        assert edge.waypoints


def test_translated_block_footprints_match_planned_centers() -> None:
    plan = plan_playtest_district()
    scenario = build_fixed_scenario()
    assert scenario.layout is not None

    for block in plan.blocks:
        footprint = scenario.layout.block_by_id(
            f"{block.block_id}_courtyard_block"
        ).footprint
        centroid = (
            sum(point[0] for point in footprint) / len(footprint),
            sum(point[1] for point in footprint) / len(footprint),
        )
        assert centroid == pytest.approx(block.plan.center)


def test_all_static_residences_are_navigation_obstacles() -> None:
    scenario = build_fixed_scenario()
    obstacles = building_obstacles(scenario, clearance=0.0)

    for building in scenario.buildings:
        point = (building.position[0], building.position[1])
        assert any(obstacle.contains(point) for obstacle in obstacles)


def test_connected_template_supports_hidden_profile_overlay() -> None:
    scenario = generate_scenario(
        seed=7,
        template_id=MAP_TEMPLATE_ID,
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )

    assert len(scenario.venues) == 36
    assert len(scenario.buildings) == 36
    assert scenario.scenario_id == f"{MAP_TEMPLATE_ID}_hp_seed_7_n2"


def test_walk_arrival_tolerance_stays_inside_venue_region() -> None:
    scenario = build_fixed_scenario()
    venue = scenario.venue_by_id("venue_red_awning_grill")
    target = meeting_target(
        1,
        venue.region.center,
        frontage_yaw_deg=venue.yaw_deg,
        agent_count=2,
    )

    radius = VenueMeetupEnv._contained_arrival_radius(target, venue, 700.0)

    expected_margin = venue.region.radius - dist(target, venue.region.center)
    assert radius == pytest.approx(expected_margin)
    assert dist(target, venue.region.center) + radius <= venue.region.radius
