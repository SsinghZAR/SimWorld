"""Regression contracts for the scalable Rosebank grid family."""

from __future__ import annotations

import math
from collections import Counter

import pytest

from benchmark.venue_meetup.building_catalog import building_bbox
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.rosebank_grid import (ROSEBANK_GRID_MAX_STEPS,
                                                  ROSEBANK_GRID_TEMPLATE_IDS,
                                                  ROSEBANK_GRID_VENUE_COUNTS,
                                                  plan_rosebank_grid)
from benchmark.venue_meetup.rosebank_roads import plan_rosebank_road_actors
from benchmark.venue_meetup.scoring import score_venue
from benchmark.venue_meetup.template_validation import collect_layout_errors
from benchmark.venue_meetup.templates import TEMPLATE_BUILDERS
from tests.venue_meetup._district_geometry_oracle import (box_clear_of_routes,
                                                          box_inside_block,
                                                          enabled_routes,
                                                          item_bounds)

_EXPECTED_LANDMARK_COUNTS = {3: 3, 5: 6, 7: 6}
_EXPECTED_ROAD_COUNTS = {
    3: {
        "carriageway": 8,
        "sidewalk": 16,
        "lane_marking": 6,
        "crosswalk": 32,
    },
    5: {
        "carriageway": 12,
        "sidewalk": 24,
        "lane_marking": 10,
        "crosswalk": 48,
    },
    7: {
        "carriageway": 16,
        "sidewalk": 32,
        "lane_marking": 12,
        "crosswalk": 64,
    },
}


def _block_centroid(block) -> tuple[float, float]:
    count = len(block.footprint)
    return tuple(
        sum(point[axis] for point in block.footprint) / count
        for axis in (0, 1)
    )


@pytest.mark.parametrize("grid_size", (3, 5, 7))
def test_scaled_templates_are_complete_city_scenarios(grid_size: int) -> None:
    template_id = ROSEBANK_GRID_TEMPLATE_IDS[grid_size]
    scenario = TEMPLATE_BUILDERS[template_id](17)

    assert scenario.map_template_id == template_id
    assert scenario.scenario_id == f"rosebank_grid_{grid_size}x{grid_size}_seed_17"
    assert scenario.layout is not None
    assert scenario.layout.layout_id == template_id
    assert len(scenario.layout.blocks) == grid_size**2
    assert len(scenario.layout.streets) == 2 * (grid_size + 1)
    assert len(scenario.venues) == ROSEBANK_GRID_VENUE_COUNTS[grid_size]
    assert len(scenario.landmarks) == _EXPECTED_LANDMARK_COUNTS[grid_size]
    assert scenario.max_steps == ROSEBANK_GRID_MAX_STEPS[grid_size]
    assert Counter(venue.zone_id for venue in scenario.venues) == {
        "zone_west": len(scenario.venues) // 2,
        "zone_east": len(scenario.venues) // 2,
    }
    assert len({venue.venue_id for venue in scenario.venues}) == len(
        scenario.venues
    )
    assert len({venue.mask_color_rgb for venue in scenario.venues}) == len(
        scenario.venues
    )

    road_counts = Counter(
        actor.kind for actor in plan_rosebank_road_actors(scenario.layout)
    )
    assert road_counts == _EXPECTED_ROAD_COUNTS[grid_size]


def test_public_map_descriptions_match_each_scale() -> None:
    compact = TEMPLATE_BUILDERS[ROSEBANK_GRID_TEMPLATE_IDS[3]](17)
    intermediate = TEMPLATE_BUILDERS[ROSEBANK_GRID_TEMPLATE_IDS[5]](17)
    large = TEMPLATE_BUILDERS[ROSEBANK_GRID_TEMPLATE_IDS[7]](17)

    assert "3x3 grid with 4 candidate venues" in compact.coarse_map_text
    assert "garden blocks" not in compact.coarse_map_text
    assert "residential edges" not in compact.coarse_map_text
    assert "4 garden blocks" in intermediate.coarse_map_text
    assert "residential edges" in large.coarse_map_text


@pytest.mark.parametrize("grid_size", (3, 5, 7))
def test_every_scaled_spawn_reaches_every_venue(grid_size: int) -> None:
    scenario = TEMPLATE_BUILDERS[ROSEBANK_GRID_TEMPLATE_IDS[grid_size]](17)
    assert scenario.layout is not None
    required_paths = [
        (agent.walk_node_id or "", frontage.approach_node_id or "")
        for agent in scenario.agents
        for frontage in scenario.layout.frontages
    ]

    assert all(start and end for start, end in required_paths)
    assert collect_layout_errors(
        scenario.layout,
        required_paths=required_paths,
    ) == []


@pytest.mark.parametrize("grid_size", (3, 5, 7))
def test_scaled_massing_stays_inside_parcels_and_clear_of_routes(
    grid_size: int,
) -> None:
    scenario = TEMPLATE_BUILDERS[ROSEBANK_GRID_TEMPLATE_IDS[grid_size]](17)
    assert scenario.layout is not None
    routes = enabled_routes(scenario.layout)

    for building in scenario.buildings:
        if not building.collision:
            continue
        block = min(
            scenario.layout.blocks,
            key=lambda candidate: math.dist(
                building.position[:2],
                _block_centroid(candidate),
            ),
        )
        bounds = item_bounds(building, building_bbox)
        assert box_inside_block(block, bounds)
        assert box_clear_of_routes(bounds, routes, clearance=200.0)


@pytest.mark.parametrize("grid_size", (3, 5, 7))
@pytest.mark.parametrize("seed", (7, 17, 31))
def test_hidden_profile_scales_with_each_city(
    grid_size: int,
    seed: int,
) -> None:
    template_id = ROSEBANK_GRID_TEMPLATE_IDS[grid_size]
    scenario = generate_scenario(
        seed=seed,
        template_id=template_id,
        num_agents=2,
        randomize=False,
        hidden_profile=True,
    )

    feasible = [
        venue
        for venue in scenario.venues
        if not score_venue(venue, scenario).hard_failures
    ]
    assert len(feasible) == 1
    assert scenario.scenario_id == f"{template_id}_hp_seed_{seed}_n2"
    assert len(scenario.venues) == ROSEBANK_GRID_VENUE_COUNTS[grid_size]
    assert {agent.zone_id for agent in scenario.agents} == {
        "zone_west",
        "zone_east",
    }


@pytest.mark.parametrize("grid_size", (1, 4, 11))
def test_grid_planner_rejects_unsupported_sizes(grid_size: int) -> None:
    with pytest.raises(ValueError, match="Unsupported Rosebank grid size"):
        plan_rosebank_grid(grid_size=grid_size)
