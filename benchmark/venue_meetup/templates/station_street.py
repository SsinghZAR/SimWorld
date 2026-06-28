"""Station-street template variant for Venue Meetup."""

from __future__ import annotations

from dataclasses import replace

from benchmark.venue_meetup.templates.central_square import build_fixed_scenario as build_central_square_scenario

MAP_TEMPLATE_ID = "station_street_v0"


def build_fixed_scenario(seed: int = 11):
    """Return a station-street flavored scenario variant."""

    scenario = build_central_square_scenario(seed)
    text = (
        "Coarse map: a straight station street with venues along the west/east sides and a hotel near the south end. "
        "The north glass hall is a public landmark near transit, while the clock-tower building marks the northwest corner. "
        "The map hides open/closed status, accessibility, crowding, food/drink, and blocked entrances."
    )
    return replace(
        scenario,
        scenario_id=f"station_street_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        coarse_map_text=text,
    )
