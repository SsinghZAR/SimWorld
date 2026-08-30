"""Focused deterministic checks for authored district visual dressing."""

from __future__ import annotations

import math

from benchmark.venue_meetup.district_dressing import (
    _DISTRICT_PROP_ASSETS,
    _DISTRICT_TREE_ASSETS,
    _SHELL_SPACING_CM,
    plan_district_actors,
    shell_positions,
)
from benchmark.venue_meetup.district_scene import DistrictSceneRenderer
from benchmark.venue_meetup.navigation import meeting_target
from benchmark.venue_meetup.scene_builder import AgentState
from benchmark.venue_meetup.templates.riverside_market import (
    build_fixed_scenario as build_large_scenario,
)
from benchmark.venue_meetup.templates.station_quarter import (
    build_fixed_scenario as build_medium_scenario,
)
from benchmark.venue_meetup.venue_env import VenueMeetupEnv
from simworld.agent.humanoid import Humanoid
from simworld.utils.vector import Vector
from tests.venue_meetup.test_scene_builder import FakeCommunicator


_ANCHOR_CLEARANCE_CM = 1_200.0
_BUILDING_CLEARANCE_CM = 3_400.0
_WALK_NODE_CLEARANCE_CM = 1_500.0
_ROUTE_CLEARANCE_CM = 1_600.0
_BRIDGE_CLEARANCE_CM = 2_200.0


def _distance(first: tuple[float, float], second: tuple[float, float]) -> float:
    return math.hypot(first[0] - second[0], first[1] - second[1])


def _segment_distance(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return _distance(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return _distance(point, (start[0] + t * dx, start[1] + t * dy))


def _minimum_route_distance(
    point: tuple[float, float],
    routes: tuple[tuple[tuple[float, float], ...], ...],
) -> float:
    return min(
        (_segment_distance(point, start, end) for route in routes for start, end in zip(route, route[1:])),
        default=math.inf,
    )


def _assert_clearance_oracle(point, scenario, layout) -> None:
    assert any(
        min(x for x, _ in block.footprint) <= point[0] <= max(x for x, _ in block.footprint)
        and min(y for _, y in block.footprint) <= point[1] <= max(y for _, y in block.footprint)
        for block in layout.blocks
    )
    assert all(_distance(point, venue.region.center) >= venue.region.radius + _ANCHOR_CLEARANCE_CM for venue in scenario.venues)
    assert all(_distance(point, venue.position[:2]) >= _BUILDING_CLEARANCE_CM for venue in scenario.venues)
    assert all(
        _distance(point, entrance.position[:2]) >= _ANCHOR_CLEARANCE_CM
        for venue in scenario.venues
        for entrance in venue.entrances
    )
    assert all(_distance(point, landmark.position[:2]) >= _BUILDING_CLEARANCE_CM for landmark in scenario.landmarks)
    nodes = tuple(node.position for node in layout.walk_nodes)
    assert all(_distance(point, node) >= _WALK_NODE_CLEARANCE_CM for node in nodes)
    routes = tuple(layout.edge_polyline(edge) for edge in layout.walk_edges)
    assert _minimum_route_distance(point, routes) >= _ROUTE_CLEARANCE_CM
    bridge_routes = tuple(
        layout.edge_polyline(edge) for edge in layout.walk_edges if edge.route_kind == "bridge"
    )
    assert _minimum_route_distance(point, bridge_routes) >= _BRIDGE_CLEARANCE_CM


def _spawn(template: str) -> tuple[FakeCommunicator, DistrictSceneRenderer]:
    scenario = build_medium_scenario(7) if template == "medium" else build_large_scenario(31)
    communicator = FakeCommunicator()
    renderer = DistrictSceneRenderer(communicator, scenario)
    renderer.spawn()
    return communicator, renderer


def _district_records(communicator: FakeCommunicator) -> list[dict[str, object]]:
    return [
        record
        for record in communicator.unrealcv.visual_spawns
        if str(record["object_name"]).startswith("GEN_BP_DISTRICT_")
    ]


def test_district_dressing_is_deterministic_and_inert() -> None:
    first, _ = _spawn("medium")
    second, _ = _spawn("medium")
    first_records = _district_records(first)
    second_records = _district_records(second)
    assert first_records == second_records
    assert len(first_records) == len({record["object_name"] for record in first_records})
    assert all(first.unrealcv.collisions[str(record["object_name"])] is False for record in first_records)
    assert all(first.unrealcv.movability[str(record["object_name"])] is False for record in first_records)


def test_district_actor_budget_preserves_medium_and_large_counts() -> None:
    assert len(plan_district_actors(build_medium_scenario(7))) == 32
    assert len(plan_district_actors(build_large_scenario(31))) == 78


def test_renderer_applies_pure_planner_order_and_locations() -> None:
    scenario = build_medium_scenario(7)
    planned = plan_district_actors(scenario)
    communicator = FakeCommunicator()
    DistrictSceneRenderer(communicator, scenario).spawn()
    records = _district_records(communicator)
    assert [record.actor_name for record in planned] == [record["object_name"] for record in records]
    assert [record.position for record in planned] == [
        tuple(communicator.unrealcv.locations[str(record["object_name"])]) for record in records
    ]


def test_props_use_allowed_assets_and_never_catalogue_roads() -> None:
    communicator, _ = _spawn("large")
    props = [
        record
        for record in _district_records(communicator)
        if str(record["object_name"]).startswith("GEN_BP_DISTRICT_PROP_")
    ]
    assert len(props) == 24
    assert all(
        not any(banned in str(record["model_path"]) for banned in ("BP_Road_C", "BP_Road1_C"))
        for record in props
    )
    assert all(
        any(asset in str(record["model_path"]) for asset in _DISTRICT_PROP_ASSETS)
        for record in props
    )


def test_medium_and_large_prop_density_differs() -> None:
    medium, _ = _spawn("medium")
    large, _ = _spawn("large")
    medium_props = [record for record in _district_records(medium) if "DISTRICT_PROP_" in str(record["object_name"])]
    large_props = [record for record in _district_records(large) if "DISTRICT_PROP_" in str(record["object_name"])]
    assert len(medium_props) == 8
    assert len(large_props) == 24
    assert len({str(record["model_path"]) for record in large_props}) > len(
        {str(record["model_path"]) for record in medium_props}
    )


def test_live_proven_trees_are_sparse_and_inert() -> None:
    for template in ("medium", "large"):
        communicator, renderer = _spawn(template)
        trees = [
            record
            for record in _district_records(communicator)
            if "DISTRICT_TREE_" in str(record["object_name"])
        ]
        assert renderer.layout is not None
        assert len(trees) == len(renderer.layout.blocks)
        assert all(
            any(asset in str(record["model_path"]) for asset in _DISTRICT_TREE_ASSETS)
            for record in trees
        )
        assert all(communicator.unrealcv.collisions[str(record["object_name"])] is False for record in trees)
        assert all(communicator.unrealcv.movability[str(record["object_name"])] is False for record in trees)


def test_every_dressing_actor_is_inside_a_block_and_clear_of_bridge_gaps() -> None:
    for template in ("medium", "large"):
        communicator, renderer = _spawn(template)
        assert renderer.layout is not None
        for record in _district_records(communicator):
            name = str(record["object_name"])
            point3d = communicator.unrealcv.locations[name]
            point = (float(point3d[0]), float(point3d[1]))
            _assert_clearance_oracle(point, renderer.scenario, renderer.layout)


def test_shells_have_conservative_cross_block_separation() -> None:
    communicator, renderer = _spawn("large")
    assert renderer.layout is not None
    shells = [
        (str(record["object_name"]), communicator.unrealcv.locations[str(record["object_name"])])
        for record in _district_records(communicator)
        if "DISTRICT_BUILDING_" in str(record["object_name"])
    ]
    for name, position in shells:
        block_id = name.removeprefix("GEN_BP_DISTRICT_BUILDING_").rsplit("_", 1)[0]
        block = renderer.layout.block_by_id(block_id)
        point = (float(position[0]), float(position[1]))
        assert min(x for x, _ in block.footprint) <= point[0] <= max(x for x, _ in block.footprint)
        assert min(y for _, y in block.footprint) <= point[1] <= max(y for _, y in block.footprint)
    for index, (_name, first) in enumerate(shells):
        for _other_name, second in shells[index + 1 :]:
            distance_sq = (float(first[0]) - float(second[0])) ** 2 + (
                float(first[1]) - float(second[1])
            ) ** 2
            assert distance_sq >= _SHELL_SPACING_CM**2


def test_route_segment_clearance_is_checked_between_nodes() -> None:
    _communicator, renderer = _spawn("medium")
    assert renderer.layout is not None
    routes = tuple(renderer.layout.edge_polyline(edge) for edge in renderer.layout.walk_edges)
    assert routes
    start, end = routes[0][0], routes[0][1]
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    assert _minimum_route_distance(midpoint, routes) < _ROUTE_CLEARANCE_CM


def test_fully_protected_block_uses_deterministic_empty_shell_policy() -> None:
    _communicator, renderer = _spawn("medium")
    assert renderer.layout is not None
    block = renderer.layout.blocks[0]
    xs, ys = zip(*block.footprint)
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    assert shell_positions(
        block,
        [],
        (),
        protected_anchors=((center, 1e9),),
        route_polylines=(),
    ) == ()


def test_meeting_target_matches_legacy_rounding_for_agent_counts() -> None:
    cases = (
        (1, 0, (0.0, 0.0), (300.0, 0.0)),
        (2, 1, (0.0, 0.0), (0.0, 300.0)),
        (3, 2, (0.0, 0.0), (-300.0, 0.0)),
        (5, 3, (123.456, -789.012), (123.46, -1089.01)),
        (5, 4, (-0.25, 0.125), (299.75, 0.12)),
    )
    for count, index, center, expected in cases:
        assert index < count
        assert meeting_target(index, center) == expected


def test_environment_meeting_target_uses_agent_order() -> None:
    scenario = build_medium_scenario(7)
    env = VenueMeetupEnv.__new__(VenueMeetupEnv)
    env.agent_ids = ["first", "middle", "last"]
    venue = scenario.venues[0]
    assert env._meeting_target("last", venue) == (-11300.0, 15150.0)


class _TeleportUnrealCV:
    def __init__(self, actor_name: str, location: tuple[float, float, float]) -> None:
        self.actor_name = actor_name
        self.locations = {actor_name: list(location)}
        self.set_locations: list[tuple[float, float, float]] = []

    def get_location(self, actor_name: str) -> list[float]:
        return list(self.locations[actor_name])

    def enable_controller(self, _actor_name: str, _enabled: int) -> None:
        return None

    def set_collision(self, _actor_name: str, _enabled: bool) -> None:
        return None

    def set_location(self, location: list[float], actor_name: str) -> None:
        self.locations[actor_name] = list(location)
        self.set_locations.append((float(location[0]), float(location[1]), float(location[2])))


class _TeleportCommunicator:
    def __init__(self, unrealcv: _TeleportUnrealCV) -> None:
        self.unrealcv = unrealcv
        self.speeds: list[tuple[int, float]] = []

    def humanoid_set_speed(self, humanoid_id: int, speed: float) -> None:
        self.speeds.append((humanoid_id, speed))


def test_teleport_navigation_places_the_shared_meeting_target(monkeypatch) -> None:
    scenario = build_medium_scenario(7)
    venue = scenario.venues[0]
    actor_name = "GEN_BP_Humanoid_teleport_test"
    unrealcv = _TeleportUnrealCV(actor_name, (0.0, 0.0, 0.0))
    communicator = _TeleportCommunicator(unrealcv)
    humanoid = Humanoid(Vector(0.0, 0.0), Vector(1.0, 0.0))
    state = AgentState("agent_2", humanoid, actor_name)
    env = VenueMeetupEnv.__new__(VenueMeetupEnv)
    env.communicator = communicator
    env.agent_ids = ["agent_0", "agent_1", "agent_2"]
    env.navigate_max_tries = 1
    env.speed = 1000.0
    env.get_agent_state = lambda _agent_id: state
    env._tick = lambda: None
    env._face_point = lambda *_args: None
    env._set_agent_walk_node_for_venue = lambda *_args: None
    monkeypatch.setattr("benchmark.venue_meetup.venue_env.time.sleep", lambda _seconds: None)

    result = env._teleport_navigate("agent_2", venue)
    expected = (-11300.0, 15150.0)
    assert unrealcv.set_locations[0][:2] == expected
    assert result["arrived"] is True
    assert result["location"] == (round(expected[0], 1), round(expected[1], 1))
