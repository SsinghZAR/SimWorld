"""Focused deterministic checks for authored district visual dressing."""

from __future__ import annotations

import math

from benchmark.venue_meetup.district_scene import (
    _DISTRICT_PROP_ASSETS,
    _DISTRICT_TREE_ASSETS,
    DistrictSceneRenderer,
)
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
_SHELL_SPACING_CM = 3_800.0


def _distance_sq(first: tuple[float, float], second: tuple[float, float]) -> float:
    return (first[0] - second[0]) ** 2 + (first[1] - second[1]) ** 2


def _segment_distance_sq(
    point: tuple[float, float],
    start: tuple[float, float],
    end: tuple[float, float],
) -> float:
    dx, dy = end[0] - start[0], end[1] - start[1]
    length_sq = dx * dx + dy * dy
    if length_sq <= 0.0:
        return _distance_sq(point, start)
    t = ((point[0] - start[0]) * dx + (point[1] - start[1]) * dy) / length_sq
    t = max(0.0, min(1.0, t))
    return _distance_sq(point, (start[0] + t * dx, start[1] + t * dy))


def _minimum_route_distance_sq(
    point: tuple[float, float],
    routes: tuple[tuple[tuple[float, float], ...], ...],
) -> float:
    return min(
        (_segment_distance_sq(point, start, end) for route in routes for start, end in zip(route, route[1:])),
        default=math.inf,
    )


def _inside_polygon(point: tuple[float, float], polygon: tuple[tuple[float, float], ...]) -> bool:
    inside = False
    for index, (x1, y1) in enumerate(polygon):
        x2, y2 = polygon[(index + 1) % len(polygon)]
        if _segment_distance_sq(point, (x1, y1), (x2, y2)) <= 1e-6:
            return True
        if (y1 > point[1]) != (y2 > point[1]):
            crossing_x = (x2 - x1) * (point[1] - y1) / (y2 - y1) + x1
            if point[0] < crossing_x:
                inside = not inside
    return inside


def _independent_anchors(scenario) -> tuple[tuple[tuple[float, float], float], ...]:
    anchors = []
    for venue in scenario.venues:
        anchors.extend(
            (
                (venue.region.center, float(venue.region.radius) + _ANCHOR_CLEARANCE_CM),
                ((venue.position[0], venue.position[1]), _BUILDING_CLEARANCE_CM),
            )
        )
        anchors.extend(
            ((entrance.position[0], entrance.position[1]), _ANCHOR_CLEARANCE_CM)
            for entrance in venue.entrances
        )
    anchors.extend(
        ((landmark.position[0], landmark.position[1]), _BUILDING_CLEARANCE_CM)
        for landmark in scenario.landmarks
    )
    return tuple(anchors)


def _assert_independent_clearance(point, scenario, layout) -> None:
    assert any(_inside_polygon(point, block.footprint) for block in layout.blocks)
    anchors = _independent_anchors(scenario)
    assert all(_distance_sq(point, anchor) >= clearance**2 for anchor, clearance in anchors)
    nodes = tuple(node.position for node in layout.walk_nodes)
    assert all(_distance_sq(point, node) >= _WALK_NODE_CLEARANCE_CM**2 for node in nodes)
    routes = tuple(
        layout.edge_polyline(edge)
        for edge in layout.walk_edges
        if len(layout.edge_polyline(edge)) >= 2
    )
    assert _minimum_route_distance_sq(point, routes) >= _ROUTE_CLEARANCE_CM**2
    bridge_routes = tuple(
        layout.edge_polyline(edge)
        for edge in layout.walk_edges
        if edge.route_kind == "bridge" and len(layout.edge_polyline(edge)) >= 2
    )
    assert _minimum_route_distance_sq(point, bridge_routes) >= _BRIDGE_CLEARANCE_CM**2


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
    medium, _ = _spawn("medium")
    large, _ = _spawn("large")
    assert len(_district_records(medium)) == 32
    assert len(_district_records(large)) == 78
    assert len(_district_records(medium)) <= 80
    assert len(_district_records(large)) <= 80


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


def test_props_clear_authored_nodes_edges_and_anchors() -> None:
    for template in ("medium", "large"):
        communicator, renderer = _spawn(template)
        assert renderer.layout is not None
        props = [
            record
            for record in _district_records(communicator)
            if "DISTRICT_PROP_" in str(record["object_name"])
        ]
        for record in props:
            position = communicator.unrealcv.locations[str(record["object_name"])]
            point = (float(position[0]), float(position[1]))
            _assert_independent_clearance(point, renderer.scenario, renderer.layout)


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
            _assert_independent_clearance(point, renderer.scenario, renderer.layout)


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
        assert _inside_polygon((float(position[0]), float(position[1])), block.footprint)
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
    assert _minimum_route_distance_sq(midpoint, routes) < _ROUTE_CLEARANCE_CM**2


def test_fully_protected_block_uses_deterministic_empty_shell_policy() -> None:
    _communicator, renderer = _spawn("medium")
    assert renderer.layout is not None
    block = renderer.layout.blocks[0]
    xs, ys = zip(*block.footprint)
    center = ((min(xs) + max(xs)) / 2.0, (min(ys) + max(ys)) / 2.0)
    assert renderer._shell_positions(
        block,
        [],
        (),
        protected_anchors=((center, 1e9),),
        route_polylines=(),
    ) == ()


def _legacy_meeting_target(center: tuple[float, float], agent_index: int) -> tuple[float, float]:
    angle = math.radians(90.0 * agent_index)
    return round(center[0] + math.cos(angle) * 300.0, 2), round(center[1] + math.sin(angle) * 300.0, 2)


def test_meeting_target_matches_legacy_rounding_for_agent_counts() -> None:
    centers = ((0.0, 0.0), (123.456, -789.012), (-0.25, 0.125))
    for center in centers:
        for count in (1, 2, 3, 5):
            for index in range(count):
                assert meeting_target(index, center) == _legacy_meeting_target(center, index)


def test_environment_meeting_target_uses_agent_order() -> None:
    scenario = build_medium_scenario(7)
    env = VenueMeetupEnv.__new__(VenueMeetupEnv)
    env.agent_ids = ["first", "middle", "last"]
    venue = scenario.venues[0]
    assert env._meeting_target("last", venue) == _legacy_meeting_target(venue.region.center, 2)


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
    expected = _legacy_meeting_target(venue.region.center, 2)
    assert unrealcv.set_locations[0][:2] == expected
    assert result["arrived"] is True
    assert result["location"] == (round(expected[0], 1), round(expected[1], 1))
