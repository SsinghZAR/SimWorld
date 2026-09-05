"""Focused contracts for the interactive busy-street playtest template."""

from __future__ import annotations

from collections import Counter

import pytest

from benchmark.venue_meetup.actions import resolve_inspect_target
from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.city_block import SIDE_OUTWARD, plan_city_block_props
from benchmark.venue_meetup.district_dressing import plan_district_actors
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.navigation import building_obstacles, plan_layout_route
from benchmark.venue_meetup.street_wall import street_wall_metrics
from benchmark.venue_meetup.template_validation import collect_layout_errors
from benchmark.venue_meetup.templates.busy_street_playtest import (
    BLOCK_SIDE_LENGTH_CM,
    MAP_TEMPLATE_ID,
    PORTAL_WIDTH_CM,
    build_fixed_scenario,
    plan_playtest_block,
    plan_playtest_street,
)


def test_playtest_block_has_unique_interactable_venue_mix() -> None:
    scenario = build_fixed_scenario(seed=17)

    assert scenario.map_template_id == MAP_TEMPLATE_ID
    assert len(scenario.venues) == 12
    assert len(scenario.buildings) == 12
    assert Counter(venue.venue_type for venue in scenario.venues) == {
        "restaurant": 6,
        "bookshop": 2,
        "bar": 2,
        "skyscraper_lobby": 2,
    }
    assert len({venue.venue_id for venue in scenario.venues}) == 12
    assert len({venue.mask_color_rgb for venue in scenario.venues}) == 12
    assert all(len(venue.entrances) == 1 for venue in scenario.venues)
    assert all(venue.visual_summary for venue in scenario.venues)
    assert {venue.zone_id for venue in scenario.venues} == {
        "zone_west",
        "zone_east",
    }
    assert Counter(venue.zone_id for venue in scenario.venues) == {
        "zone_west": 6,
        "zone_east": 6,
    }


def test_restaurants_resolve_individually_by_id_and_name() -> None:
    scenario = build_fixed_scenario()
    restaurants = [
        venue for venue in scenario.venues if venue.venue_type == "restaurant"
    ]

    assert len(restaurants) == 6
    for venue in restaurants:
        assert (
            resolve_inspect_target(
                scenario.venues,
                venue.region.center,
                target_venue_id=venue.venue_id,
            )
            is venue
        )
        display_name = venue.visual_summary.split(":", 1)[0]
        assert (
            resolve_inspect_target(
                scenario.venues,
                venue.region.center,
                target_description=display_name,
            )
            is venue
        )


def test_playtest_block_has_four_continuous_sides_and_clear_portals() -> None:
    plan = plan_playtest_block()
    buildings = plan.buildings

    assert len(buildings) == 24
    assert plan.side_length_cm == BLOCK_SIDE_LENGTH_CM
    assert plan.portal_width_cm == PORTAL_WIDTH_CM
    assert {portal.side for portal in plan.portals} == {
        "north",
        "east",
        "south",
        "west",
    }
    assert all(
        portal.conservative_clear_width_cm >= 800.0
        for portal in plan.portals
    )
    assert sorted(building.placement.index for building in buildings) == list(
        range(24)
    )
    for side in ("north", "east", "south", "west"):
        assert len(plan.buildings_on_side(side)) == 6
    for run in plan.runs:
        metrics = street_wall_metrics(
            tuple(building.placement for building in run.buildings),
            run.length_cm,
        )
        assert metrics.coverage == pytest.approx(1.0)
        assert metrics.maximum_gap_cm == pytest.approx(0.0, abs=1e-6)


def test_playtest_block_keeps_props_out_of_portal_spans() -> None:
    plan = plan_playtest_block()
    props = plan_city_block_props(plan)

    assert [prop.index for prop in props] == list(range(len(props)))
    for prop in props:
        side = plan.side_for_building(prop.building_index)
        outward = SIDE_OUTWARD[side]
        tangent = (-outward[1], outward[0])
        along = (
            (prop.position[0] - plan.center[0]) * tangent[0]
            + (prop.position[1] - plan.center[1]) * tangent[1]
        )
        assert abs(along) >= PORTAL_WIDTH_CM / 2.0


def test_skyscrapers_are_tallest_buildings_in_block() -> None:
    buildings = plan_playtest_street()
    physical_heights = {
        building.placement.index: (
            building_bbox(building.placement.asset_key)[2]
            * building.placement.scale[2]
        )
        for building in buildings
    }
    skyscraper_heights = [
        physical_heights[building.placement.index]
        for building in buildings
        if building.use == "skyscraper_lobby"
    ]
    other_heights = [
        physical_heights[building.placement.index]
        for building in buildings
        if building.use != "skyscraper_lobby"
    ]
    assert min(skyscraper_heights) > max(other_heights)


def test_authored_sidewalk_reaches_every_venue_without_generic_shells() -> None:
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    required_paths = [
        (agent.walk_node_id, f"approach_{venue.slot_id}")
        for agent in scenario.agents
        for venue in scenario.venues
    ]
    required_paths.extend(
        (agent.walk_node_id, f"portal_{side}_inner")
        for agent in scenario.agents
        for side in ("north", "east", "south", "west")
    )

    assert collect_layout_errors(
        scenario.layout,
        required_paths=required_paths,
    ) == []
    assert plan_district_actors(scenario) == ()
    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        for venue in scenario.venues:
            route = plan_layout_route(
                scenario.layout,
                agent.walk_node_id,
                venue_slot_id=venue.slot_id,
            )
            assert route is not None
            assert route.access_path[-1] == venue.region.center


def test_each_portal_connects_outer_sidewalk_to_courtyard_ring() -> None:
    scenario = build_fixed_scenario()
    assert scenario.layout is not None

    for side in ("north", "east", "south", "west"):
        outer_id = f"portal_{side}_outer"
        threshold_id = f"portal_{side}_threshold"
        inner_id = f"portal_{side}_inner"
        courtyard_id = f"courtyard_{side}"
        direct_edges = {
            frozenset((edge.start_node_id, edge.end_node_id)): edge
            for edge in scenario.layout.walk_edges
        }
        for left, right in (
            (outer_id, threshold_id),
            (threshold_id, inner_id),
            (inner_id, courtyard_id),
        ):
            edge = direct_edges[frozenset((left, right))]
            assert edge.route_kind == "alley"
            assert edge.waypoints

    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        for side in ("north", "east", "south", "west"):
            path = scenario.layout.shortest_path(
                agent.walk_node_id,
                f"courtyard_{side}",
            )
            assert path is not None
            assert any(node_id.endswith("_threshold") for node_id in path)


def test_static_residences_are_navigation_obstacles() -> None:
    scenario = build_fixed_scenario()
    obstacles = building_obstacles(scenario, clearance=0.0)

    for building in scenario.buildings:
        point = (building.position[0], building.position[1])
        assert any(obstacle.contains(point) for obstacle in obstacles)


def test_playtest_template_supports_hidden_profile_overlay() -> None:
    scenario = generate_scenario(
        seed=7,
        template_id=MAP_TEMPLATE_ID,
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )

    assert len(scenario.venues) == 12
    assert len(scenario.buildings) == 12
    assert scenario.scenario_id == f"{MAP_TEMPLATE_ID}_hp_seed_7_n2"
