"""Canal/bridge template variant for Venue Meetup."""

from __future__ import annotations

from dataclasses import replace

from benchmark.venue_meetup.templates.central_square import build_fixed_scenario as build_central_square_scenario

MAP_TEMPLATE_ID = "canal_bridge_v0"


def build_fixed_scenario(seed: int = 13):
    """Return a canal/bridge flavored scenario variant."""

    scenario = build_central_square_scenario(seed)
    text = (
        "Coarse map: a canal-like north/south divider crosses the district with candidate venues around a bridge-like central crossing. "
        "The clock-tower building and hospital-ramp building are the main localization landmarks. "
        "The map hides open/closed status, accessibility, crowding, food/drink, and blocked entrances."
    )
    return replace(
        scenario,
        scenario_id=f"canal_bridge_seed_{seed}",
        map_template_id=MAP_TEMPLATE_ID,
        coarse_map_text=text,
    )
