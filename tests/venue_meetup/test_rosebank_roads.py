"""Contracts for the Rosebank visual road-dressing plan."""

from __future__ import annotations

from collections import Counter
from dataclasses import replace

from benchmark.venue_meetup.rosebank_roads import (
    ASPHALT_ASSET_KEY,
    MARKING_ASSET_KEY,
    SIDEWALK_ASSET_KEY,
    plan_rosebank_road_actors,
)
from benchmark.venue_meetup.templates.rosebank_grid_playtest import (
    build_district_layout,
    build_fixed_scenario,
)


def test_road_plan_adds_complete_hierarchical_street_dressing() -> None:
    scenario = build_fixed_scenario()
    actors = plan_rosebank_road_actors(scenario.layout)

    assert len(actors) == 152
    assert Counter(actor.kind for actor in actors) == {
        "carriageway": 20,
        "sidewalk": 40,
        "lane_marking": 12,
        "crosswalk": 80,
    }
    assert len({actor.actor_id for actor in actors}) == len(actors)
    assert all(actor.actor_id.startswith("GEN_BP_ROAD_") for actor in actors)
    assert all(not actor.collision and not actor.movable for actor in actors)
    assert {actor.asset_key for actor in actors} == {
        ASPHALT_ASSET_KEY,
        SIDEWALK_ASSET_KEY,
        MARKING_ASSET_KEY,
    }
    assert all("BP_Road1" not in actor.asset_path for actor in actors)


def test_road_plan_is_scoped_to_the_rosebank_layout() -> None:
    layout = build_district_layout()

    assert plan_rosebank_road_actors(None) == ()
    assert plan_rosebank_road_actors(
        replace(layout, layout_id="unrelated_layout")
    ) == ()
