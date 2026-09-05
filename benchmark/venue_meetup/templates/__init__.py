"""Built-in venue-meetup templates."""

from .busy_street_playtest import (
    build_fixed_scenario as build_busy_street_playtest_scenario,
)
from .canal_bridge import build_fixed_scenario as build_canal_bridge_scenario
from .central_square import build_fixed_scenario as build_central_square_scenario
from .connected_blocks_playtest import (
    build_fixed_scenario as build_connected_blocks_playtest_scenario,
)
from .riverside_market import build_fixed_scenario as build_riverside_market_scenario
from .station_quarter import build_fixed_scenario as build_station_quarter_scenario
from .station_street import build_fixed_scenario as build_station_street_scenario

TEMPLATE_BUILDERS = {
    "central_square_v0": build_central_square_scenario,
    "station_street_v0": build_station_street_scenario,
    "canal_bridge_v0": build_canal_bridge_scenario,
    "busy_street_playtest_v0": build_busy_street_playtest_scenario,
    "connected_blocks_playtest_v0": build_connected_blocks_playtest_scenario,
    "station_quarter_medium_v1": build_station_quarter_scenario,
    "riverside_market_large_v1": build_riverside_market_scenario,
}

__all__ = [
    "TEMPLATE_BUILDERS",
    "build_canal_bridge_scenario",
    "build_busy_street_playtest_scenario",
    "build_central_square_scenario",
    "build_connected_blocks_playtest_scenario",
    "build_riverside_market_scenario",
    "build_station_quarter_scenario",
    "build_station_street_scenario",
]
