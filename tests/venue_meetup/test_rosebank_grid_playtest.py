"""Contracts for the Rosebank-inspired 9x9 mixed-use playtest."""

from __future__ import annotations

import math
from collections import Counter
from itertools import combinations

import pytest

from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.district_dressing import plan_district_actors
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.navigation import building_obstacles, plan_layout_route
from benchmark.venue_meetup.rosebank_grid import (
    ALLEY_FRONTAGE_OFFSET_CM,
    BLOCK_PITCH_CM,
    BLOCK_SIDE_CM,
    GARDEN_BLOCK_IDS,
    GRID_SIZE,
    LANDMARK_BLOCK_ROLES,
)
from benchmark.venue_meetup.rosebank_grid_layout import (
    alley_center_node_id,
    venue_geometry,
    vertical_mid_node_id,
)
from benchmark.venue_meetup.template_validation import collect_layout_errors
from benchmark.venue_meetup.templates.rosebank_grid_playtest import (
    MAP_TEMPLATE_ID,
    build_fixed_scenario,
    plan_playtest_grid,
)


def _world_half_extents(building) -> tuple[float, float]:
    raw_x, raw_y, _raw_z = building_bbox(building.asset_key)
    half_x = raw_x * building.scale[0] / 2.0
    half_y = raw_y * building.scale[1] / 2.0
    radians = math.radians(building.yaw_deg)
    cosine, sine = abs(math.cos(radians)), abs(math.sin(radians))
    return (
        cosine * half_x + sine * half_y,
        sine * half_x + cosine * half_y,
    )


def _segment_hits_box(
    start: tuple[float, float],
    end: tuple[float, float],
    bounds: tuple[float, float, float, float],
) -> bool:
    """Return whether a finite route segment intersects an axis-aligned box."""

    min_x, min_y, max_x, max_y = bounds
    lower_t, upper_t = 0.0, 1.0
    for origin, delta, lower, upper in (
        (start[0], end[0] - start[0], min_x, max_x),
        (start[1], end[1] - start[1], min_y, max_y),
    ):
        if math.isclose(delta, 0.0, abs_tol=1e-9):
            if not lower <= origin <= upper:
                return False
            continue
        first = (lower - origin) / delta
        second = (upper - origin) / delta
        axis_lower, axis_upper = sorted((first, second))
        lower_t = max(lower_t, axis_lower)
        upper_t = min(upper_t, axis_upper)
        if lower_t > upper_t:
            return False
    return True


def test_plan_is_nine_by_nine_with_mixed_zoning_and_street_hierarchy() -> None:
    plan = plan_playtest_grid()

    assert GRID_SIZE == 9
    assert len(plan.blocks) == 81
    assert len({block.block_id for block in plan.blocks}) == 81
    assert plan.block_at(0, 0).block_id == "A1"
    assert plan.block_at(8, 8).block_id == "I9"
    assert len(plan.street_x) == len(plan.street_y) == 10
    assert plan.street_x[1] - plan.street_x[0] == BLOCK_PITCH_CM
    assert Counter(block.zone for block in plan.blocks) == {
        "mixed_use": 53,
        "residential": 12,
        "transit_core": 7,
        "civic": 5,
        "garden": 4,
    }
    assert {
        block.block_id for block in plan.blocks if block.zone == "garden"
    } == set(GARDEN_BLOCK_IDS)
    assert {
        block.block_id: block.landmark_role
        for block in plan.blocks
        if block.landmark_role is not None
    } == LANDMARK_BLOCK_ROLES


def test_alley_shortcuts_cross_blocks_and_join_the_public_grid() -> None:
    plan = plan_playtest_grid()
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    layout = scenario.layout

    assert Counter(axis for block in plan.blocks for axis in block.alley_axes) == {
        "horizontal": 21,
        "vertical": 16,
    }
    assert sum(edge.route_kind == "alley" for edge in layout.walk_edges) == 74

    block = plan.block_by_id("C3")
    start = vertical_mid_node_id(block.column, block.row)
    end = vertical_mid_node_id(block.column + 1, block.row)
    path = layout.shortest_path(start, end)
    assert path == [start, alley_center_node_id("C3"), end]
    assert layout.path_length_cm(start, end) == pytest.approx(BLOCK_PITCH_CM)


def test_venues_are_offset_from_crossing_alley_openings() -> None:
    plan = plan_playtest_grid()
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    venue_by_id = {venue.venue_id: venue for venue in scenario.venues}
    alley_segments = [
        (left, right)
        for edge in scenario.layout.walk_edges
        if edge.route_kind == "alley"
        for left, right in zip(
            scenario.layout.edge_polyline(edge),
            scenario.layout.edge_polyline(edge)[1:],
        )
    ]

    for site in plan.venue_sites:
        block = plan.block_by_id(site.block_id)
        crosses_alley_opening = (
            "vertical" in block.alley_axes
            and site.side in {"north", "south"}
        ) or (
            "horizontal" in block.alley_axes
            and site.side in {"east", "west"}
        )
        expected_offset = ALLEY_FRONTAGE_OFFSET_CM if crosses_alley_opening else 0.0
        assert abs(site.frontage_offset_cm) == expected_offset

        venue = venue_by_id[site.venue_id]
        geometry = venue_geometry(plan, site)
        assert venue.position[:2] == geometry.position
        half_x, half_y = _world_half_extents(venue)
        bounds = (
            venue.position[0] - half_x - 200.0,
            venue.position[1] - half_y - 200.0,
            venue.position[0] + half_x + 200.0,
            venue.position[1] + half_y + 200.0,
        )
        assert not any(
            _segment_hits_box(start, end, bounds)
            for start, end in alley_segments
        )


def test_scenario_has_unique_mixed_use_venues_and_landmark_anchors() -> None:
    scenario = build_fixed_scenario()

    assert scenario.map_template_id == MAP_TEMPLATE_ID
    assert len(scenario.venues) == 36
    assert len(scenario.landmarks) == 6
    assert Counter(venue.zone_id for venue in scenario.venues) == {
        "zone_west": 18,
        "zone_east": 18,
    }
    assert Counter(venue.venue_type for venue in scenario.venues) == {
        "restaurant": 5,
        "cafe": 5,
        "shop": 5,
        "bookshop": 5,
        "bar": 4,
        "hotel_lobby": 4,
        "skyscraper_lobby": 4,
        "pub": 4,
    }
    assert len({venue.venue_id for venue in scenario.venues}) == 36
    assert len({venue.slot_id for venue in scenario.venues}) == 36
    colors = [
        *(venue.mask_color_rgb for venue in scenario.venues),
        *(landmark.mask_color_rgb for landmark in scenario.landmarks),
    ]
    assert len(set(colors)) == 42
    for left, right in combinations(colors, 2):
        assert max(abs(left[index] - right[index]) for index in range(3)) > 8
        assert max(abs(left[index] - right[2 - index]) for index in range(3)) > 8


def test_every_agent_reaches_every_frontage_on_valid_graph() -> None:
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    required_paths: list[tuple[str, str]] = []
    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        for frontage in scenario.layout.frontages:
            assert frontage.approach_node_id is not None
            required_paths.append((agent.walk_node_id, frontage.approach_node_id))

    assert collect_layout_errors(
        scenario.layout,
        required_paths=required_paths,
    ) == []
    assert all(
        scenario.layout.is_reachable(start, end)
        for start, end in required_paths
    )
    route = plan_layout_route(
        scenario.layout,
        scenario.agents[1].walk_node_id or "",
        venue_slot_id=scenario.venues[0].slot_id,
    )
    assert route is not None
    assert route.graph_distance_cm > 40_000.0


def test_bounded_massing_stays_inside_blocks_and_clear_of_routes() -> None:
    plan = plan_playtest_grid()
    scenario = build_fixed_scenario()
    assert scenario.layout is not None
    assert 350 <= len(scenario.buildings) <= 500
    assert sum(building.collision for building in scenario.buildings) == 379
    assert sum(not building.collision for building in scenario.buildings) == 51
    assert plan_district_actors(scenario) == ()

    route_segments = [
        (left, right)
        for edge in scenario.layout.walk_edges
        for left, right in zip(
            scenario.layout.edge_polyline(edge),
            scenario.layout.edge_polyline(edge)[1:],
        )
    ]
    half_block = BLOCK_SIDE_CM / 2.0
    for building in scenario.buildings:
        nearest = min(
            plan.blocks,
            key=lambda block: math.dist(block.center, building.position[:2]),
        )
        dx = abs(building.position[0] - nearest.center[0])
        dy = abs(building.position[1] - nearest.center[1])
        assert dx <= half_block and dy <= half_block
        if not building.collision:
            continue
        half_x, half_y = _world_half_extents(building)
        assert dx + half_x <= half_block + 1e-6
        assert dy + half_y <= half_block + 1e-6
        bounds = (
            building.position[0] - half_x - 200.0,
            building.position[1] - half_y - 200.0,
            building.position[0] + half_x + 200.0,
            building.position[1] + half_y + 200.0,
        )
        assert not any(
                _segment_hits_box(start, end, bounds)
            for start, end in route_segments
        )


def test_collision_disabled_trees_are_not_navigation_obstacles() -> None:
    scenario = build_fixed_scenario()
    solid_count = sum(building.collision for building in scenario.buildings)
    obstacles = building_obstacles(scenario, clearance=0.0)

    assert len(obstacles) == (
        len(scenario.venues) + len(scenario.landmarks) + solid_count
    )


def test_hidden_profile_overlay_preserves_city_geometry() -> None:
    scenario = generate_scenario(
        seed=23,
        template_id=MAP_TEMPLATE_ID,
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )

    assert len(scenario.layout.blocks) == 81
    assert len(scenario.venues) == 36
    assert scenario.scenario_id == f"{MAP_TEMPLATE_ID}_hp_seed_23_n2"
