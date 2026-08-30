"""Focused deterministic checks for authored district visual dressing."""

from __future__ import annotations

from benchmark.venue_meetup.district_scene import (
    _DISTRICT_PROP_ASSETS,
    _DISTRICT_TREE_ASSETS,
    _SHELL_SPACING_CM,
    DistrictSceneRenderer,
)
from benchmark.venue_meetup.templates.riverside_market import (
    build_fixed_scenario as build_large_scenario,
)
from benchmark.venue_meetup.templates.station_quarter import (
    build_fixed_scenario as build_medium_scenario,
)
from tests.venue_meetup.test_scene_builder import FakeCommunicator


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
        nodes = tuple(node.position for node in renderer.layout.walk_nodes)
        props = [
            record
            for record in _district_records(communicator)
            if "DISTRICT_PROP_" in str(record["object_name"])
        ]
        for record in props:
            position = communicator.unrealcv.locations[str(record["object_name"])]
            point = (float(position[0]), float(position[1]))
            assert renderer._clear(
                point,
                renderer._protected_anchors(),
                nodes,
                renderer._route_polylines(),
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
        blocks = renderer.layout.blocks
        nodes = tuple(node.position for node in renderer.layout.walk_nodes)
        routes = renderer._route_polylines()
        bridge_routes = renderer._bridge_gap_polylines()
        for record in _district_records(communicator):
            name = str(record["object_name"])
            point3d = communicator.unrealcv.locations[name]
            point = (float(point3d[0]), float(point3d[1]))
            assert any(renderer._inside_block(block, point) for block in blocks)
            assert renderer._clear(
                point,
                renderer._protected_anchors(),
                nodes,
                routes,
                bridge_polylines=bridge_routes,
            )


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
        assert renderer._inside_block(block, (float(position[0]), float(position[1])))
    for index, (_name, first) in enumerate(shells):
        for _other_name, second in shells[index + 1 :]:
            distance_sq = (float(first[0]) - float(second[0])) ** 2 + (
                float(first[1]) - float(second[1])
            ) ** 2
            assert distance_sq >= _SHELL_SPACING_CM**2


def test_route_segment_clearance_is_checked_between_nodes() -> None:
    _communicator, renderer = _spawn("medium")
    routes = renderer._route_polylines()
    assert routes
    start, end = routes[0][0], routes[0][1]
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    assert not renderer._clear(midpoint, (), (), routes)


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
