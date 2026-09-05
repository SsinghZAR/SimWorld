"""Scene lifecycle helpers for Venue Meetup (clear, light, spawn, settle)."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from itertools import permutations
from typing import Any, Mapping, Sequence

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.district_scene import DistrictSceneRenderer
from benchmark.venue_meetup.rosebank_roads import plan_rosebank_road_actors
from benchmark.venue_meetup.scenario import Scenario
from simworld.agent.humanoid import Humanoid
from simworld.communicator.communicator import Communicator
from simworld.communicator.unrealcv import BlueprintObjectSpec
from simworld.config import Config
from simworld.utils.vector import Vector

AGENT_BLUEPRINT = "/Game/TrafficSystem/Pedestrian/Base_User_Agent.Base_User_Agent_C"
BATCH_SPAWN_THRESHOLD = 96


@dataclass
class AgentState:
    """Runtime mapping from benchmark agent id to SimWorld humanoid actor."""

    agent_id: str
    humanoid: Humanoid
    actor_name: str


def direction_from_yaw(yaw: float) -> Vector:
    """Build a unit Vector from a yaw angle."""

    return Vector(math.cos(math.radians(yaw)), math.sin(math.radians(yaw))).normalize()


def _xy(value: Any) -> tuple[float, float]:
    """Normalize a numeric sequence or UnrealCV coordinate string."""

    if isinstance(value, bytes):
        value = value.decode("utf-8")
    coordinates = value.split() if isinstance(value, str) else value
    return float(coordinates[0]), float(coordinates[1])


class SceneBuilder:
    """Owns UE scene clear/light/spawn/settle for a Venue Meetup episode."""

    def __init__(
        self,
        communicator: Communicator,
        scenario: Scenario,
        *,
        config: Config | None = None,
        resolution: tuple[int, int] = (640, 360),
        speed: float = 1000.0,
        agent_blueprint: str = AGENT_BLUEPRINT,
        tick_count: int = 1,
        spawn_settle_sec: float = 0.7,
        sun_rotation: tuple[float, float, float] = (-50.0, 180.0, 180.0),
    ) -> None:
        self.communicator = communicator
        self.scenario = scenario
        self.config = config or Config()
        self.resolution = resolution
        self.speed = speed
        self.agent_blueprint = agent_blueprint
        self.tick_count = tick_count
        self.spawn_settle_sec = spawn_settle_sec
        self.sun_rotation = sun_rotation
        self._sun_actor: str | None = None

    def prepare_environment(self) -> None:
        """Enter async mode, wipe prior GEN_BP actors, and reset humanoid ids."""

        # Async (running) mode so the engine character movement actually walks and
        # is blocked by building collision. Sync/pause only advances on manual
        # ticks, under which the packaged StepForward does not progress.
        self.communicator.unrealcv.set_mode("async")
        self.communicator.clear_env(keep_roads=True)
        Humanoid._id_counter = 0
        Humanoid._camera_id_counter = 1

    def setup_lighting(self) -> None:
        """Force a deterministic, well-lit sun on the otherwise dim empty map.

        The packaged empty map ships with a low/odd sun angle and no usable
        weather-manager blueprint, so the only reliable lever is rotating the
        scene's existing directional light. Re-applying it every reset also
        undoes any orientation left behind by earlier episodes.
        """

        if self._sun_actor is None:
            try:
                objects = list(self.communicator.unrealcv.get_objects())
            except Exception:  # noqa: BLE001 - lighting is best-effort.
                objects = []
            self._sun_actor = next((name for name in objects if name.lower().startswith("directionallight")), "")
        if not self._sun_actor:
            return
        try:
            self.communicator.unrealcv.set_orientation(self.sun_rotation, self._sun_actor)
        except Exception:  # noqa: BLE001 - never fail an episode over lighting.
            pass

    def spawn_static_scene(self) -> None:
        """Spawn authored city dressing, venues, landmarks, and visible props."""

        DistrictSceneRenderer(self.communicator, self.scenario).spawn()
        specs = self._blueprint_object_specs()
        batch_spawn = getattr(
            self.communicator.unrealcv,
            "spawn_bp_assets_batch",
            None,
        )
        if callable(batch_spawn) and len(specs) >= BATCH_SPAWN_THRESHOLD:
            batch_spawn(specs)
            return
        self._spawn_static_scene_sequential(specs)

    def _blueprint_object_specs(self) -> tuple[BlueprintObjectSpec, ...]:
        """Flatten scenario actors into complete engine setup records."""

        specs = [
            BlueprintObjectSpec(
                prefab_path=actor.asset_path,
                name=actor.actor_id,
                location=actor.position,
                rotation=(0.0, actor.yaw_deg, 0.0),
                scale=actor.scale,
                collision=actor.collision,
                movable=actor.movable,
            )
            for actor in plan_rosebank_road_actors(self.scenario.layout)
        ]
        for building in self.scenario.buildings:
            specs.append(
                BlueprintObjectSpec(
                    prefab_path=building.asset_path,
                    name=self.building_actor_name(building.building_id),
                    location=building.position,
                    rotation=(0.0, building.yaw_deg, 0.0),
                    scale=building.scale,
                    collision=building.collision,
                    movable=False,
                )
            )
        for venue in self.scenario.venues:
            specs.append(
                BlueprintObjectSpec(
                    prefab_path=venue.asset_path,
                    name=self.venue_actor_name(venue.venue_id),
                    location=venue.position,
                    rotation=(0.0, venue.yaw_deg, 0.0),
                    scale=venue.scale,
                    color=venue.mask_color_rgb,
                )
            )
            specs.extend(
                BlueprintObjectSpec(
                    prefab_path=asset_path(prop.asset_key),
                    name=self.prop_actor_name(prop.prop_id),
                    location=prop.position,
                    rotation=(0.0, prop.yaw_deg, 0.0),
                    scale=prop.scale,
                    color=prop.color_rgb,
                )
                for prop in venue.props
            )
        specs.extend(
            BlueprintObjectSpec(
                prefab_path=landmark.asset_path,
                name=self.landmark_actor_name(landmark.landmark_id),
                location=landmark.position,
                rotation=(0.0, landmark.yaw_deg, 0.0),
                scale=landmark.scale,
                color=landmark.mask_color_rgb,
            )
            for landmark in self.scenario.landmarks
        )
        return tuple(specs)

    def _spawn_static_scene_sequential(
        self,
        specs: Sequence[BlueprintObjectSpec],
    ) -> None:
        """Apply the compatibility path used by small scenes and test adapters."""

        for spec in specs:
            self.communicator.spawn_object(
                spec.name,
                spec.prefab_path,
                spec.location,
                spec.rotation,
                scale=spec.scale,
            )
            if spec.color is not None:
                self.communicator.unrealcv.set_color(spec.name, spec.color)
            self.communicator.unrealcv.set_collision(
                spec.name,
                spec.collision,
            )
            self.communicator.unrealcv.set_movable(spec.name, spec.movable)

    def spawn_agents(self) -> dict[str, AgentState]:
        """Spawn all humanoid agents and return their runtime states."""

        agent_states: dict[str, AgentState] = {}
        for agent in self.scenario.agents:
            direction = direction_from_yaw(agent.yaw_deg)
            humanoid = Humanoid(
                position=Vector(agent.position[0], agent.position[1]),
                direction=direction,
                communicator=self.communicator,
                config=self.config,
            )
            self.communicator.spawn_agent(
                humanoid,
                name=None,
                position=agent.position,
                model_path=self.agent_blueprint,
                type="humanoid",
            )
            actor_name = self.communicator.get_humanoid_name(humanoid.id)
            self.communicator.unrealcv.set_orientation((0.0, agent.yaw_deg, 0.0), actor_name)
            self.communicator.humanoid_set_speed(humanoid.id, self.speed)
            agent_states[agent.agent_id] = AgentState(
                agent_id=agent.agent_id,
                humanoid=humanoid,
                actor_name=actor_name,
            )

        self._match_cameras_to_agents(agent_states)

        # Configure cameras only after every humanoid exists.  Spawning a later
        # humanoid can reset earlier camera defaults, so this final deterministic
        # pass makes the requested resolution authoritative for all agents.
        for agent in self.scenario.agents:
            self.communicator.unrealcv.set_camera_resolution(
                agent_states[agent.agent_id].humanoid.camera_id,
                self.resolution,
            )
        return agent_states

    def _match_cameras_to_agents(
        self,
        agent_states: Mapping[str, AgentState],
    ) -> None:
        """Match sensor IDs to pawns when UE registers cameras out of order."""

        states = tuple(agent_states.values())
        if len(states) < 2:
            return
        camera_ids = tuple(state.humanoid.camera_id for state in states)
        try:
            actor_positions = tuple(
                _xy(self.communicator.unrealcv.get_location(state.actor_name))
                for state in states
            )
            camera_positions = {
                camera_id: _xy(
                    self.communicator.unrealcv.get_camera_location(camera_id)
                )
                for camera_id in camera_ids
            }
        except (AttributeError, IndexError, KeyError, TypeError, ValueError):
            # Some offline adapters and older packages do not expose camera
            # poses. Their existing sequential IDs remain the best fallback.
            return

        assignment = min(
            permutations(camera_ids),
            key=lambda candidate: sum(
                math.dist(actor_position, camera_positions[camera_id])
                for actor_position, camera_id in zip(actor_positions, candidate)
            ),
        )
        for state, camera_id in zip(states, assignment):
            state.humanoid.camera_id = camera_id

    def settle(self, agent_states: Mapping[str, AgentState], agent_ids: Sequence[str]) -> None:
        """Let freshly spawned agents drop to the ground and come to rest."""

        for agent_id in agent_ids:
            try:
                self.communicator.humanoid_stop(agent_states[agent_id].humanoid.id)
            except Exception:  # noqa: BLE001 - settling is best-effort.
                pass
        time.sleep(self.spawn_settle_sec)
        self.tick()

    def tick(self) -> None:
        """Advance the engine by the configured tick count."""

        for _ in range(max(1, self.tick_count)):
            self.communicator.unrealcv.tick()

    @staticmethod
    def venue_actor_name(venue_id: str) -> str:
        """Return deterministic UE actor name for a venue.

        The ``GEN_BP_`` prefix lets ``clear_env(keep_roads=True)`` wipe every
        scene actor on the next reset; otherwise venues/landmarks/props would
        accumulate and collide across episodes and scenarios.
        """

        return f"GEN_BP_VENUE_{venue_id}"

    @staticmethod
    def building_actor_name(building_id: str) -> str:
        """Return a deterministic actor name for non-interactive buildings."""

        return f"GEN_BP_BUILDING_{building_id}"

    @staticmethod
    def landmark_actor_name(landmark_id: str) -> str:
        """Return deterministic UE actor name for a landmark."""

        return f"GEN_BP_LANDMARK_{landmark_id}"

    @staticmethod
    def prop_actor_name(prop_id: str) -> str:
        """Return deterministic UE actor name for a prop."""

        return f"GEN_BP_PROP_{prop_id}"
