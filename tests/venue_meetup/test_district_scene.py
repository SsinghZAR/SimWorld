"""Focused deterministic checks for authored district visual dressing."""

from __future__ import annotations

from benchmark.venue_meetup.district_scene import (
    _DISTRICT_PROP_ASSETS,
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
