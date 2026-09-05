"""Offline fake-communicator tests for Venue Meetup SceneBuilder."""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any
import pytest

from benchmark.venue_meetup.building_catalog import asset_path
from benchmark.venue_meetup.district_dressing import plan_district_actors
from benchmark.venue_meetup.templates.station_quarter import build_fixed_scenario as build_station_scenario
from benchmark.venue_meetup.scenario import (
    AgentSpec,
    Landmark,
    PropSpec,
    Region,
    Requirement,
    Scenario,
    StaticBuilding,
    Venue,
    VenueProperties,
)
from benchmark.venue_meetup.scene_builder import AGENT_BLUEPRINT, AgentState, SceneBuilder, direction_from_yaw
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.agent.humanoid import Humanoid
from simworld.utils.vector import Vector


class FakeUnrealCV:
    """In-memory UnrealCV stub that records command order without sleeping."""

    def __init__(
        self,
        *,
        objects: list[str] | None = None,
        camera_locations: dict[int, tuple[float, float, float]] | None = None,
        fail_get_objects: bool = False,
        fail_set_orientation: bool = False,
        log: list[tuple[Any, ...]] | None = None,
    ) -> None:
        self.objects = list(objects) if objects is not None else ["DirectionalLight_0"]
        self.camera_locations = dict(camera_locations or {})
        self.fail_get_objects = fail_get_objects
        self.fail_set_orientation = fail_set_orientation
        self.log = log if log is not None else []
        self.orientations: dict[str, tuple[float, float, float]] = {}
        self.colors: dict[str, tuple[int, int, int]] = {}
        self.scales: dict[str, tuple[float, float, float]] = {}
        self.collisions: dict[str, bool] = {}
        self.movability: dict[str, bool] = {}
        self.visual_spawns: list[dict[str, Any]] = []
        self.locations: dict[str, Any] = {}
        self.camera_resolutions: dict[Any, tuple[int, int]] = {}
        self.camera_resolution_history: list[tuple[Any, tuple[int, int]]] = []
        self.modes: list[str] = []
        self.ticks = 0

    def set_mode(self, mode: str, *args: Any) -> None:
        self.modes.append(mode)
        self.log.append(("set_mode", mode, *args))

    def get_objects(self) -> list[str]:
        self.log.append(("get_objects",))
        if self.fail_get_objects:
            raise RuntimeError("get_objects failed")
        return list(self.objects)

    def set_orientation(self, orientation: tuple[float, float, float], actor_name: str) -> None:
        self.log.append(("set_orientation", orientation, actor_name))
        if self.fail_set_orientation:
            raise RuntimeError("set_orientation failed")
        self.orientations[actor_name] = orientation

    def set_color(self, actor_name: str, color_rgb: tuple[int, int, int]) -> None:
        self.log.append(("set_color", actor_name, color_rgb))
        self.colors[actor_name] = color_rgb

    def set_scale(self, scale: tuple[float, float, float], actor_name: str) -> None:
        self.log.append(("set_scale", scale, actor_name))
        self.scales[actor_name] = scale

    def spawn_bp_asset(self, model_path: str, actor_name: str) -> None:
        self.log.append(("spawn_bp_asset", model_path, actor_name))
        self.visual_spawns.append({"object_name": actor_name, "model_path": model_path})

    def set_location(self, location: Any, actor_name: str) -> None:
        self.log.append(("set_location", location, actor_name))
        self.locations[actor_name] = location

    def set_collision(self, actor_name: str, has_collision: bool) -> None:
        self.log.append(("set_collision", actor_name, has_collision))
        self.collisions[actor_name] = has_collision

    def set_movable(self, actor_name: str, is_movable: bool) -> None:
        self.log.append(("set_movable", actor_name, is_movable))
        self.movability[actor_name] = is_movable

    def set_camera_resolution(self, camera_id: Any, resolution: tuple[int, int]) -> None:
        self.log.append(("set_camera_resolution", camera_id, resolution))
        self.camera_resolution_history.append((camera_id, resolution))
        self.camera_resolutions[camera_id] = resolution

    def get_location(self, actor_name: str) -> Any:
        return self.locations[actor_name]

    def get_camera_location(self, camera_id: int) -> tuple[float, float, float]:
        return self.camera_locations[camera_id]

    def tick(self) -> None:
        self.log.append(("tick",))
        self.ticks += 1


class BatchFakeUnrealCV(FakeUnrealCV):
    """Fake that exposes the optimized large-scene batch API."""

    def __init__(self) -> None:
        super().__init__()
        self.batch_specs: tuple[Any, ...] = ()

    def spawn_bp_assets_batch(self, specs: Any) -> list[str]:
        self.batch_specs = tuple(specs)
        self.log.append(("spawn_bp_assets_batch", len(self.batch_specs)))
        return ["ok"] * len(self.batch_specs)


class FakeCommunicator:
    """Communicator stub used by SceneBuilder / VenueMeetupEnv offline tests."""

    def __init__(self, unrealcv: FakeUnrealCV | None = None) -> None:
        self.log: list[tuple[Any, ...]] = []
        if unrealcv is None:
            self.unrealcv = FakeUnrealCV(log=self.log)
        else:
            self.unrealcv = unrealcv
            self.unrealcv.log = self.log
        self.spawned_objects: list[dict[str, Any]] = []
        self.spawned_agents: list[dict[str, Any]] = []
        self.humanoid_speeds: dict[int, float] = {}
        self.humanoid_stops: list[int] = []
        self.humanoid_id_to_name: dict[int, str] = {}
        self.clear_calls: list[dict[str, Any]] = []

    def clear_env(self, keep_roads: bool = False) -> None:
        self.clear_calls.append({"keep_roads": keep_roads})
        self.log.append(("clear_env", keep_roads))

    def spawn_object(
        self,
        object_name: str,
        model_path: str,
        position: Any,
        direction: Any,
        scale: tuple[float, float, float] = (1.0, 1.0, 1.0),
    ) -> None:
        record = {
            "object_name": object_name,
            "model_path": model_path,
            "position": position,
            "direction": direction,
            "scale": scale,
        }
        self.spawned_objects.append(record)
        self.log.append(("spawn_object", object_name, model_path, position, direction))

    def spawn_agent(self, agent: Any, name: Any, position: Any = None, model_path: str = "", type: str = "humanoid") -> None:
        record = {
            "agent": agent,
            "name": name,
            "position": position,
            "model_path": model_path,
            "type": type,
        }
        self.spawned_agents.append(record)
        self.log.append(("spawn_agent", agent.id, name, position, model_path, type))
        actor_name = f"GEN_BP_Humanoid_{agent.id}"
        self.humanoid_id_to_name[agent.id] = actor_name
        self.unrealcv.locations[actor_name] = position

    def get_humanoid_name(self, humanoid_id: int) -> str:
        if humanoid_id not in self.humanoid_id_to_name:
            self.humanoid_id_to_name[humanoid_id] = f"GEN_BP_Humanoid_{humanoid_id}"
        return self.humanoid_id_to_name[humanoid_id]

    def humanoid_set_speed(self, humanoid_id: int, speed: float) -> None:
        self.humanoid_speeds[humanoid_id] = speed
        self.log.append(("humanoid_set_speed", humanoid_id, speed))

    def humanoid_stop(self, humanoid_id: int) -> None:
        self.humanoid_stops.append(humanoid_id)
        self.log.append(("humanoid_stop", humanoid_id))


def _tiny_scenario() -> Scenario:
    props = [
        PropSpec(
            prop_id="cone_a",
            asset_key="RoadCone_C",
            position=(15.0, 25.0, 0.0),
            yaw_deg=30.0,
            scale=(1.5, 1.5, 2.0),
            color_rgb=(10, 20, 30),
        ),
        PropSpec(
            prop_id="blocker_b",
            asset_key="RoadBlocker_C",
            position=(16.0, 26.0, 0.0),
            yaw_deg=0.0,
            scale=(1.0, 1.0, 1.0),
            color_rgb=None,
        ),
    ]
    venue = Venue(
        venue_id="venue_cafe",
        slot_id="slot_cafe",
        venue_type="cafe",
        asset_key="BP_Building_05_C",
        asset_path="/Game/Fake/Venue.Venue_C",
        position=(100.0, 200.0, 0.0),
        yaw_deg=45.0,
        region=Region(center=(100.0, 200.0), radius=500.0),
        mask_color_rgb=(255, 0, 0),
        properties=VenueProperties(
            open=True,
            reachable=True,
            capacity=4,
            accessible=True,
            shelter=True,
            food_drink=True,
            quiet_score=0.5,
            crowding_score=0.2,
        ),
        entrances=[],
        props=props,
        visual_summary="red cafe",
    )
    landmark = Landmark(
        landmark_id="clock",
        slot_id="slot_clock",
        landmark_type="clock_tower",
        asset_key="BP_Building_20_C",
        asset_path="/Game/Fake/Clock.Clock_C",
        position=(0.0, 50.0, 0.0),
        yaw_deg=90.0,
        mask_color_rgb=(0, 255, 0),
        visual_summary="clock tower",
    )
    agents = [
        AgentSpec(
            agent_id="alice",
            spawn_slot="spawn_a",
            position=(1.0, 2.0, 100.0),
            yaw_deg=90.0,
            private_constraint="quiet",
            private_requirement_keys=["quiet"],
        ),
        AgentSpec(
            agent_id="bob",
            spawn_slot="spawn_b",
            position=(3.0, 4.0, 100.0),
            yaw_deg=0.0,
            private_constraint="food",
            private_requirement_keys=["food_drink"],
        ),
    ]
    building = StaticBuilding(
        building_id="row_house",
        asset_key="BP_Building_01_C",
        asset_path="/Game/Fake/House.House_C",
        position=(-100.0, 200.0, 0.0),
        yaw_deg=45.0,
        scale=(0.5, 0.5, 0.75),
    )
    return Scenario(
        scenario_id="scene_builder_test",
        map_template_id="tiny",
        seed=1,
        venues=[venue],
        landmarks=[landmark],
        agents=agents,
        requirements=[Requirement(key="open", weight=1.0, hard=True)],
        soft_weights={"quiet": 1.0},
        coarse_map_text="tiny map",
        max_steps=8,
        buildings=[building],
    )


def _builder(
    communicator: FakeCommunicator,
    scenario: Scenario | None = None,
    **kwargs: Any,
) -> SceneBuilder:
    return SceneBuilder(
        communicator,  # type: ignore[arg-type]
        scenario or _tiny_scenario(),
        config=SimpleNamespace(),  # type: ignore[arg-type]
        **kwargs,
    )


def test_static_scene_actor_names_path_pose_color_scale() -> None:
    communicator = FakeCommunicator()
    scenario = _tiny_scenario()
    builder = _builder(communicator, scenario)

    builder.spawn_static_scene()

    venue = scenario.venues[0]
    landmark = scenario.landmarks[0]
    cone, blocker = venue.props
    by_name = {item["object_name"]: item for item in communicator.spawned_objects}

    assert set(by_name) == {
        SceneBuilder.building_actor_name(scenario.buildings[0].building_id),
        SceneBuilder.venue_actor_name(venue.venue_id),
        SceneBuilder.prop_actor_name(cone.prop_id),
        SceneBuilder.prop_actor_name(blocker.prop_id),
        SceneBuilder.landmark_actor_name(landmark.landmark_id),
    }

    building = scenario.buildings[0]
    building_name = SceneBuilder.building_actor_name(building.building_id)
    building_spawn = by_name[building_name]
    assert building_spawn["model_path"] == building.asset_path
    assert building_spawn["scale"] == building.scale
    assert communicator.unrealcv.collisions[building_name] is True
    assert communicator.unrealcv.movability[building_name] is False

    venue_spawn = by_name[SceneBuilder.venue_actor_name(venue.venue_id)]
    assert venue_spawn["model_path"] == venue.asset_path
    assert venue_spawn["position"] == venue.position
    assert venue_spawn["direction"] == (0.0, venue.yaw_deg, 0.0)
    assert communicator.unrealcv.colors[
        SceneBuilder.venue_actor_name(venue.venue_id)
    ] == venue.mask_color_rgb

    cone_spawn = by_name[SceneBuilder.prop_actor_name(cone.prop_id)]
    assert cone_spawn["model_path"] == asset_path(cone.asset_key)
    assert cone_spawn["position"] == cone.position
    assert cone_spawn["direction"] == (0.0, cone.yaw_deg, 0.0)
    assert communicator.unrealcv.scales[
        SceneBuilder.prop_actor_name(cone.prop_id)
    ] == cone.scale
    assert communicator.unrealcv.colors[
        SceneBuilder.prop_actor_name(cone.prop_id)
    ] == cone.color_rgb

    blocker_name = SceneBuilder.prop_actor_name(blocker.prop_id)
    assert communicator.unrealcv.scales[blocker_name] == blocker.scale
    assert blocker_name not in communicator.unrealcv.colors

    landmark_spawn = by_name[SceneBuilder.landmark_actor_name(landmark.landmark_id)]
    assert landmark_spawn["model_path"] == landmark.asset_path
    assert landmark_spawn["position"] == landmark.position
    assert landmark_spawn["direction"] == (0.0, landmark.yaw_deg, 0.0)
    assert communicator.unrealcv.colors[
        SceneBuilder.landmark_actor_name(landmark.landmark_id)
    ] == landmark.mask_color_rgb


def test_large_static_scene_uses_complete_batch_specs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    unrealcv = BatchFakeUnrealCV()
    communicator = FakeCommunicator(unrealcv)
    scenario = _tiny_scenario()
    monkeypatch.setattr(
        "benchmark.venue_meetup.scene_builder.BATCH_SPAWN_THRESHOLD",
        1,
    )

    _builder(communicator, scenario).spawn_static_scene()

    specs = unrealcv.batch_specs
    assert len(specs) == 5
    by_name = {spec.name: spec for spec in specs}
    building = scenario.buildings[0]
    building_spec = by_name[SceneBuilder.building_actor_name(building.building_id)]
    assert building_spec.prefab_path == building.asset_path
    assert building_spec.location == building.position
    assert building_spec.collision is True
    assert building_spec.movable is False
    venue = scenario.venues[0]
    venue_spec = by_name[SceneBuilder.venue_actor_name(venue.venue_id)]
    assert venue_spec.color == venue.mask_color_rgb
    assert venue_spec.scale == venue.scale
    assert ("spawn_bp_assets_batch", 5) in communicator.log
    assert communicator.spawned_objects == []


def test_layout_backed_scene_spawns_inert_city_dressing_from_authored_geometry() -> None:
    communicator = FakeCommunicator()
    scenario = build_station_scenario(seed=7)

    _builder(communicator, scenario).spawn_static_scene()

    dressing = communicator.unrealcv.visual_spawns
    names = [item["object_name"] for item in dressing]
    planned = plan_district_actors(scenario)
    assert dressing
    assert names == [record.actor_name for record in planned]
    assert len(names) == len(set(names))
    assert any(name.startswith("GEN_BP_DISTRICT_BUILDING_") for name in names)
    # Solid shells back the authored city geometry; decorative props and trees
    # remain inert so they do not perturb navigation clearance.
    assert all(
        communicator.unrealcv.collisions[name] is name.startswith("GEN_BP_DISTRICT_BUILDING_")
        for name in names
    )
    assert all(communicator.unrealcv.movability[name] is False for name in names)


def test_spawn_agents_builds_state_orientation_speed_and_camera(monkeypatch: pytest.MonkeyPatch) -> None:
    communicator = FakeCommunicator()
    scenario = _tiny_scenario()
    resolution = (320, 180)
    speed = 750.0
    builder = _builder(communicator, scenario, resolution=resolution, speed=speed)
    Humanoid._id_counter = 0
    Humanoid._camera_id_counter = 1

    agent_states = builder.spawn_agents()

    assert list(agent_states) == ["alice", "bob"]
    for agent in scenario.agents:
        state = agent_states[agent.agent_id]
        assert isinstance(state, AgentState)
        assert state.agent_id == agent.agent_id
        assert state.actor_name == f"GEN_BP_Humanoid_{state.humanoid.id}"
        assert state.humanoid.position == Vector(agent.position[0], agent.position[1])
        expected_dir = direction_from_yaw(agent.yaw_deg)
        assert abs(float(state.humanoid.direction.x) - expected_dir.x) < 1e-9
        assert abs(float(state.humanoid.direction.y) - expected_dir.y) < 1e-9
        assert communicator.unrealcv.orientations[state.actor_name] == (0.0, agent.yaw_deg, 0.0)
        assert communicator.humanoid_speeds[state.humanoid.id] == speed
        assert communicator.unrealcv.camera_resolutions[state.humanoid.camera_id] == resolution

    resolution_calls = [
        (index, call[1], call[2])
        for index, call in enumerate(communicator.log)
        if call[0] == "set_camera_resolution"
    ]
    last_spawn_index = max(
        index for index, call in enumerate(communicator.log) if call[0] == "spawn_agent"
    )
    assert [camera_id for _index, camera_id, _resolution in resolution_calls] == [
        agent_states[agent.agent_id].humanoid.camera_id for agent in scenario.agents
    ]
    assert all(index > last_spawn_index for index, _camera_id, _resolution in resolution_calls)
    assert communicator.unrealcv.camera_resolution_history == [
        (camera_id, resolution) for _index, camera_id, resolution in resolution_calls
    ]

    assert [record["model_path"] for record in communicator.spawned_agents] == [AGENT_BLUEPRINT, AGENT_BLUEPRINT]
    assert [record["position"] for record in communicator.spawned_agents] == [
        scenario.agents[0].position,
        scenario.agents[1].position,
    ]


def test_spawn_agents_matches_reversed_engine_camera_order() -> None:
    scenario = _tiny_scenario()
    first, second = scenario.agents
    unrealcv = FakeUnrealCV(
        camera_locations={
            1: second.position,
            2: first.position,
        }
    )
    communicator = FakeCommunicator(unrealcv)
    Humanoid._id_counter = 0
    Humanoid._camera_id_counter = 1

    agent_states = _builder(communicator, scenario).spawn_agents()

    assert agent_states[first.agent_id].humanoid.camera_id == 2
    assert agent_states[second.agent_id].humanoid.camera_id == 1
    assert communicator.unrealcv.camera_resolution_history == [
        (2, (640, 360)),
        (1, (640, 360)),
    ]


def test_reset_lifecycle_ordering(monkeypatch: pytest.MonkeyPatch) -> None:
    sleep_calls: list[float] = []
    monkeypatch.setattr("benchmark.venue_meetup.scene_builder.time.sleep", lambda seconds: sleep_calls.append(seconds))

    communicator = FakeCommunicator()
    scenario = _tiny_scenario()
    env = VenueMeetupEnv(
        communicator,  # type: ignore[arg-type]
        scenario,
        config=SimpleNamespace(),  # type: ignore[arg-type]
        spawn_settle_sec=0.7,
        tick_count=2,
        sun_rotation=(-50.0, 180.0, 180.0),
    )
    monkeypatch.setattr(env, "_build_observations", lambda: {agent_id: {"agent_id": agent_id} for agent_id in env.agent_ids})

    observations = env.reset()

    assert observations == {"alice": {"agent_id": "alice"}, "bob": {"agent_id": "bob"}}
    assert env.spawned is True
    assert sleep_calls == [0.7]
    assert communicator.unrealcv.modes == ["async"]
    assert communicator.clear_calls == [{"keep_roads": True}]
    assert list(env.agent_states) == ["alice", "bob"]

    markers: list[str] = []
    for call in communicator.log:
        name = call[0]
        if name == "set_mode":
            markers.append("async")
        elif name == "clear_env":
            markers.append("clear")
        elif name == "get_objects":
            markers.append("lighting")
        elif name == "spawn_object" and str(call[1]).startswith("GEN_BP_VENUE_"):
            markers.append("static")
        elif name == "spawn_agent":
            markers.append("agents")
        elif name == "humanoid_stop":
            markers.append("settle_stop")
        elif name == "tick":
            markers.append("tick")

    assert markers.index("async") < markers.index("clear")
    assert markers.index("clear") < markers.index("lighting")
    assert markers.index("lighting") < markers.index("static")
    assert markers.index("static") < markers.index("agents")
    assert markers.index("agents") < markers.index("settle_stop")
    assert markers.index("settle_stop") < markers.index("tick")
    assert communicator.unrealcv.ticks == 2
    assert communicator.unrealcv.orientations["DirectionalLight_0"] == (-50.0, 180.0, 180.0)


def test_lighting_best_effort_get_objects_failure() -> None:
    communicator = FakeCommunicator(FakeUnrealCV(fail_get_objects=True))
    builder = _builder(communicator)

    builder.setup_lighting()

    assert ("get_objects",) in communicator.log
    assert not any(call[0] == "set_orientation" for call in communicator.log)


def test_lighting_best_effort_set_orientation_failure() -> None:
    communicator = FakeCommunicator(FakeUnrealCV(fail_set_orientation=True))
    builder = _builder(communicator)

    builder.setup_lighting()

    assert ("get_objects",) in communicator.log
    assert any(call[0] == "set_orientation" for call in communicator.log)


def test_lighting_skips_when_no_directional_light() -> None:
    communicator = FakeCommunicator(FakeUnrealCV(objects=["SkyAtmosphere_0"]))
    builder = _builder(communicator)

    builder.setup_lighting()

    assert builder._sun_actor == ""
    assert all(call[0] != "set_orientation" for call in communicator.log)
