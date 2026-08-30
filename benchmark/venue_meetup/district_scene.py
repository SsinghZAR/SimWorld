"""UE adapter for deterministic visual dressing of authored districts.

Geometry, candidate selection, clearances, and actor ordering live in the pure
``district_dressing`` planner.  This module only applies those records through
the UnrealCV communicator.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.district_dressing import (
    DistrictActorRecord,
    building_actor_name as _building_actor_name,
    plan_district_actors,
)

if TYPE_CHECKING:
    from benchmark.venue_meetup.scenario import Scenario
    from simworld.communicator.communicator import Communicator


class DistrictSceneRenderer:
    """Spawn the pure planner's static, inert district actors."""

    def __init__(self, communicator: Communicator, scenario: Scenario) -> None:
        self.communicator = communicator
        self.scenario = scenario
        self.layout = scenario.layout

    def spawn(self) -> None:
        """Apply planned actor records in deterministic order."""

        for record in plan_district_actors(self.scenario):
            self._spawn_decor(record)

    def _spawn_decor(self, record: DistrictActorRecord) -> None:
        unrealcv = self.communicator.unrealcv
        unrealcv.spawn_bp_asset(asset_path(record.asset_key), record.actor_name)
        unrealcv.set_location(record.position, record.actor_name)
        unrealcv.set_orientation((0.0, record.yaw_deg, 0.0), record.actor_name)
        unrealcv.set_scale(record.scale, record.actor_name)
        unrealcv.set_collision(record.actor_name, False)
        unrealcv.set_movable(record.actor_name, False)

    @staticmethod
    def building_actor_name(block_id: str, shell_index: int) -> str:
        return _building_actor_name(block_id, shell_index)
