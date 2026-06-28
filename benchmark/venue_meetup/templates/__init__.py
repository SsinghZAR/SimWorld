"""Built-in venue-meetup templates."""

from .canal_bridge import build_fixed_scenario as build_canal_bridge_scenario
from .central_square import build_fixed_scenario as build_central_square_scenario
from .station_street import build_fixed_scenario as build_station_street_scenario

TEMPLATE_BUILDERS = {
    "central_square_v0": build_central_square_scenario,
    "station_street_v0": build_station_street_scenario,
    "canal_bridge_v0": build_canal_bridge_scenario,
}

__all__ = [
    "TEMPLATE_BUILDERS",
    "build_canal_bridge_scenario",
    "build_central_square_scenario",
    "build_station_street_scenario",
]
