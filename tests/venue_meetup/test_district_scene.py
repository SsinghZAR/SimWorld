"""Focused deterministic checks for authored district visual dressing."""

from __future__ import annotations

import json
import math

import pytest

from benchmark.venue_meetup import building_catalog
from benchmark.venue_meetup.district_dressing import (
    _DISTRICT_PROP_ASSETS,
    _DISTRICT_TREE_ASSETS,
    plan_district_actors,
    plan_shell_records,
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
from tests.venue_meetup._district_geometry_oracle import (
    box_clear_of_routes,
    box_inside_block,
    boxes_separate,
    distance,
    edge_frames,
    edge_projection,
    enabled_bridge_routes,
    enabled_routes,
    fixture_gaps,
    frontage_edge,
    item_bounds,
    measured_half_extents,
    minimum_route_distance,
    point_in_polygon,
    shell_edge_metrics,
)


_ANCHOR_CLEARANCE_CM = 1_200.0
_BUILDING_CLEARANCE_CM = 3_400.0
_WALK_NODE_CLEARANCE_CM = 1_500.0
_ROUTE_CLEARANCE_CM = 1_600.0
_BRIDGE_CLEARANCE_CM = 2_200.0


# Test-owned authored gap fixtures.  Keeping the endpoint/frontage/bridge
# intervals here makes coverage independent of the tiler's private predicates.
_GAP_FIXTURES = {
    "station_quarter_medium_v1": {
        "end_gap_cm": 2_500.0,
        "frontages": {
            "block_nw": ("front_nw_market_cafe", "front_nw_cross_bistro"),
            "block_ne": ("front_ne_market_shop", "front_ne_cross_deli"),
            "block_sw": ("front_sw_cross_pub", "front_sw_alley_lobby"),
            "block_se": ("front_se_cross_hall", "front_se_alley_market"),
        },
        "bridge_edges": (),
        "gaps": {
            "block_nw": {
                0: ((0.0, 2_500.0), (7_000.0, 9_000.0), (12_500.0, 15_000.0)),
                1: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
                2: ((0.0, 2_500.0), (8_000.0, 10_000.0), (12_500.0, 15_000.0)),
                3: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
            },
            "block_ne": {
                0: ((0.0, 2_500.0), (6_000.0, 8_000.0), (12_500.0, 15_000.0)),
                1: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
                2: ((0.0, 2_500.0), (4_973.207162891235, 7_026.792837108765), (12_500.0, 15_000.0)),
                3: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
            },
            "block_sw": {
                0: ((0.0, 2_500.0), (4_694.999816894531, 7_305.000183105469), (12_500.0, 15_000.0)),
                1: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
                2: ((0.0, 2_500.0), (6_000.0, 8_000.0), (12_500.0, 15_000.0)),
                3: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
            },
            "block_se": {
                0: ((0.0, 2_500.0), (6_500.0, 11_500.0), (12_500.0, 15_000.0)),
                1: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
                2: ((0.0, 2_500.0), (6_197.610107421875, 9_802.389892578125), (12_500.0, 15_000.0)),
                3: ((0.0, 2_500.0), (9_500.0, 12_000.0)),
            },
        },
    },
    "riverside_market_large_v1": {
        "end_gap_cm": 2_500.0,
        "frontages": {
            "block_nw_civic": ("front_nw_civic_cafe", "front_nw_civic_shop"),
            "block_w_market": ("front_w_market_bistro", "front_w_market_stall"),
            "block_sw_residential": ("front_sw_resid_pub", "front_sw_resid_lobby"),
            "block_ne_transit": ("front_ne_transit_shop", "front_ne_transit_deli"),
            "block_e_waterfront": ("front_e_water_restaurant", "front_e_water_hall"),
            "block_se_hotel": ("front_se_hotel_cafe", "front_se_hotel_shop"),
        },
        "bridge_edges": (
            ("bridge_primary_west", "bridge_primary_east"),
            ("bridge_secondary_west", "bridge_secondary_east"),
        ),
        "gaps": {
            "block_nw_civic": {
                0: ((0.0, 2_500.0), (19_000.0, 21_000.0), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
                2: ((0.0, 2_500.0), (19_000.0, 21_000.0), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
            },
            "block_w_market": {
                0: ((0.0, 2_500.0), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (4_973.207162891235, 7_026.792837108765), (8_000.0, 10_000.0), (11_500.0, 14_000.0)),
                2: ((0.0, 2_500.0), (17_000.0, 19_000.0), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (11_500.0, 14_000.0)),
            },
            "block_sw_residential": {
                0: ((0.0, 2_500.0), (18_694.99981689453, 21_305.00018310547), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (6_000.0, 8_000.0), (17_500.0, 20_000.0)),
                2: ((0.0, 2_500.0), (17_000.0, 19_000.0), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
            },
            "block_ne_transit": {
                0: ((0.0, 2_500.0), (7_000.0, 9_000.0), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
                2: ((0.0, 2_500.0), (6_973.207162891235, 9_026.792837108766), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
            },
            "block_e_waterfront": {
                0: ((0.0, 2_500.0), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (11_500.0, 14_000.0)),
                2: ((0.0, 2_500.0), (9_000.0, 11_000.0), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (4_000.0, 6_000.0), (6_197.610107421875, 9_802.389892578125), (11_500.0, 14_000.0)),
            },
            "block_se_hotel": {
                0: ((0.0, 2_500.0), (5_500.0, 10_500.0), (25_500.0, 28_000.0)),
                1: ((0.0, 2_500.0), (17_500.0, 20_000.0)),
                2: ((0.0, 2_500.0), (9_000.0, 11_000.0), (25_500.0, 28_000.0)),
                3: ((0.0, 2_500.0), (12_000.0, 14_000.0), (17_500.0, 20_000.0)),
            },
        },
    },
}

_EXPECTED_LAYOUT_METRICS = {
    "station_quarter_medium_v1": {
        "shells": 64,
        "per_block": 16,
        "minimum_scale": 0.18,
        "eligible_cm": 114_731.63417441529,
        "coverage": 0.6789118810521176,
        "max_gap_cm": 831.359230070324,
    },
    "riverside_market_large_v1": {
        "shells": 144,
        "per_block": 24,
        "minimum_scale": 0.18,
        "eligible_cm": 394_678.04850019776,
        "coverage": 0.5257885234079842,
        "max_gap_cm": 2_034.4806912367458,
    },
}


def test_bounding_boxes_normalize_positive_numeric_extents(tmp_path, monkeypatch) -> None:
    path = tmp_path / "bounding_boxes.json"
    path.write_text(
        json.dumps({"buildings": {"fixture": {"bbox": {"x": 1, "y": 2.5, "z": 3}}}}),
        encoding="utf-8",
    )
    monkeypatch.setattr(building_catalog, "BOUNDING_BOXES_PATH", path)
    building_catalog.bounding_boxes.cache_clear()
    try:
        assert building_catalog.bounding_boxes() == {
            "buildings": {"fixture": {"bbox": {"x": 1.0, "y": 2.5, "z": 3.0}}}
        }
        assert building_catalog.building_bbox("fixture") == (1.0, 2.5, 3.0)
    finally:
        building_catalog.bounding_boxes.cache_clear()


@pytest.mark.parametrize(
    "payload",
    (
        {"buildings": {"fixture": []}},
        {"buildings": {"fixture": {"bbox": {"x": 1, "y": 2}}}},
        {"buildings": {"fixture": {"bbox": {"x": 1, "y": 2, "z": 3, "w": 4}}}},
        {"buildings": {"fixture": {"bbox": {"x": "1", "y": 2, "z": 3}}}},
        {"buildings": {"fixture": {"bbox": {"x": True, "y": 2, "z": 3}}}},
        {"buildings": {"fixture": {"bbox": {"x": 0, "y": 2, "z": 3}}}},
        {"buildings": {"fixture": {"bbox": {"x": float("nan"), "y": 2, "z": 3}}}},
        {"buildings": {"fixture": {"bbox": {"x": float("inf"), "y": 2, "z": 3}}}},
    ),
)
def test_bounding_boxes_reject_malformed_records(payload, tmp_path, monkeypatch) -> None:
    path = tmp_path / "bounding_boxes.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    monkeypatch.setattr(building_catalog, "BOUNDING_BOXES_PATH", path)
    building_catalog.bounding_boxes.cache_clear()
    try:
        with pytest.raises(ValueError, match="(?i)bounding"):
            building_catalog.bounding_boxes()
    finally:
        building_catalog.bounding_boxes.cache_clear()


def _assert_clearance_oracle(point, scenario, layout) -> None:
    assert any(point_in_polygon(point, block.footprint) for block in layout.blocks)
    assert all(distance(point, venue.region.center) >= venue.region.radius + _ANCHOR_CLEARANCE_CM for venue in scenario.venues)
    assert all(distance(point, venue.position[:2]) >= _BUILDING_CLEARANCE_CM for venue in scenario.venues)
    assert all(
        distance(point, entrance.position[:2]) >= _ANCHOR_CLEARANCE_CM
        for venue in scenario.venues
        for entrance in venue.entrances
    )
    assert all(distance(point, landmark.position[:2]) >= _BUILDING_CLEARANCE_CM for landmark in scenario.landmarks)
    nodes = tuple(node.position for node in layout.walk_nodes)
    assert all(distance(point, node) >= _WALK_NODE_CLEARANCE_CM for node in nodes)
    routes = enabled_routes(layout)
    assert minimum_route_distance(point, routes) >= _ROUTE_CLEARANCE_CM
    assert minimum_route_distance(point, enabled_bridge_routes(layout)) >= _BRIDGE_CLEARANCE_CM

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
    for record in first_records:
        name = str(record["object_name"])
        # Shells are the only solid district actors; props and trees remain
        # visual-only so they cannot perturb the authored walk graph.
        expected_collision = "DISTRICT_BUILDING_" in name
        assert first.unrealcv.collisions[name] is expected_collision
    assert all(first.unrealcv.movability[str(record["object_name"])] is False for record in first_records)


def test_district_actor_budget_preserves_medium_and_large_counts() -> None:
    for scenario, expected_total, expected_shells in (
        (build_medium_scenario(7), 76, 64),
        (build_large_scenario(31), 174, 144),
    ):
        actors = plan_district_actors(scenario)
        assert len(actors) == expected_total
        assert sum(record.footprint is not None for record in actors) == expected_shells


def test_shell_coverage_uses_only_eligible_edge_intervals() -> None:
    for scenario in (build_medium_scenario(7), build_large_scenario(31)):
        layout = scenario.layout
        assert layout is not None
        expected = _EXPECTED_LAYOUT_METRICS[layout.layout_id]
        records = plan_shell_records(scenario)
        assert len(records) == expected["shells"]
        assert min(min(record.scale) for record in records) >= 0.18
        assert min(min(record.scale) for record in records) == pytest.approx(expected["minimum_scale"])
        metrics = shell_edge_metrics(
            scenario, _GAP_FIXTURES, plan_shell_records
        )
        assert metrics["eligible_cm"] == pytest.approx(expected["eligible_cm"], rel=1e-8)
        assert metrics["coverage"] == pytest.approx(expected["coverage"], rel=1e-8)
        assert metrics["max_gap_cm"] == pytest.approx(expected["max_gap_cm"], rel=1e-8)
        assert metrics["coverage"] >= (0.55 if expected["shells"] == 64 else 0.50)
        assert metrics["max_gap_cm"] <= (2_000.0 if expected["shells"] == 64 else 3_500.0)
        per_block = metrics["per_block"]
        assert all(item[0] == expected["per_block"] for item in per_block.values())


def test_shell_footprints_match_measured_asset_envelopes() -> None:
    """Recompute every shell AABB from its public asset metadata independently."""

    for scenario in (build_medium_scenario(7), build_large_scenario(31)):
        records = plan_shell_records(scenario)
        assert records
        for record in records:
            footprint = record.footprint
            assert footprint is not None
            expected = measured_half_extents(record, building_catalog.building_bbox)
            assert footprint.half_extents[0] == pytest.approx(expected[0], abs=1e-6)
            assert footprint.half_extents[1] == pytest.approx(expected[1], abs=1e-6)


def test_undeclared_gap_changes_the_eligible_denominator_and_coverage() -> None:
    scenario = build_medium_scenario(7)
    baseline = shell_edge_metrics(
        scenario, _GAP_FIXTURES, plan_shell_records
    )
    # This gap is deliberately not part of the explicit frontage/end/bridge
    # fixture.  A hidden exclusion must not silently disappear from metrics.
    tampered = shell_edge_metrics(
        scenario,
        _GAP_FIXTURES,
        plan_shell_records,
        extra_gaps=(("block_nw", 0, 9_500.0, 10_500.0),),
    )
    assert tampered["eligible_cm"] < baseline["eligible_cm"]
    assert tampered["coverage"] != pytest.approx(baseline["coverage"], rel=1e-8)
    expected = _EXPECTED_LAYOUT_METRICS[scenario.layout.layout_id]  # type: ignore[union-attr]
    assert tampered["eligible_cm"] != pytest.approx(expected["eligible_cm"], rel=1e-8)
    assert tampered["coverage"] != pytest.approx(expected["coverage"], rel=1e-8)


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
        shell_records = {
            record.actor_name: record
            for record in plan_shell_records(renderer.scenario)
        }
        protected_bounds = tuple(
            item_bounds(item, building_catalog.building_bbox, 200.0)
            for item in (*renderer.scenario.venues, *renderer.scenario.landmarks)
        )
        routes = enabled_routes(renderer.layout)
        bridge_routes = enabled_bridge_routes(renderer.layout)
        for record in _district_records(communicator):
            name = str(record["object_name"])
            point3d = communicator.unrealcv.locations[name]
            point = (float(point3d[0]), float(point3d[1]))
            shell = shell_records.get(name)
            if shell is None:
                _assert_clearance_oracle(point, renderer.scenario, renderer.layout)
                continue
            footprint = shell.footprint
            assert footprint is not None
            block = renderer.layout.block_by_id(footprint.block_id)
            assert box_inside_block(block, footprint.bounds)
            assert all(boxes_separate(footprint.bounds, protected) for protected in protected_bounds)
            assert box_clear_of_routes(footprint.bounds, routes, 400.0)
            assert box_clear_of_routes(footprint.bounds, bridge_routes, 500.0)


def test_declared_frontage_end_and_bridge_gap_fixtures_are_explicit() -> None:
    for scenario in (build_medium_scenario(7), build_large_scenario(31)):
        layout = scenario.layout
        assert layout is not None
        fixture = _GAP_FIXTURES[layout.layout_id]
        fixture_frontages = fixture["frontages"]
        all_frontage_ids = {
            frontage.frontage_id
            for frontage in layout.frontages
            if frontage.block_id in fixture_frontages
        }
        assert all_frontage_ids == {
            frontage_id
            for frontage_ids in fixture_frontages.values()
            for frontage_id in frontage_ids
        }
        for block in layout.blocks:
            expected_ids = set(fixture_frontages[block.block_id])
            actual_ids = {
                frontage.frontage_id
                for frontage in layout.frontages
                if frontage.block_id == block.block_id
            }
            assert actual_ids == expected_ids
            for edge in edge_frames(block):
                gaps = fixture_gaps(scenario, block, edge, _GAP_FIXTURES)
                assert gaps[0][0] == pytest.approx(0.0)
                assert gaps[-1][1] == pytest.approx(edge.length)
                assert gaps[0][1] >= fixture["end_gap_cm"] - 1e-6
                assert gaps[-1][0] <= edge.length - fixture["end_gap_cm"] + 1e-6
                for frontage in layout.frontages:
                    if frontage.block_id != block.block_id:
                        continue
                    assigned = frontage_edge(block, frontage)
                    if assigned.index != edge.index:
                        continue
                    along, _normal = edge_projection(frontage.position[:2], edge)
                    assert any(left <= along <= right for left, right in gaps)

        bridge_edges = {
            frozenset((edge.start_node_id, edge.end_node_id))
            for edge in layout.walk_edges
            if edge.route_kind == "bridge"
        }
        assert bridge_edges == {frozenset(edge_ids) for edge_ids in fixture["bridge_edges"]}
        if bridge_edges:
            assert all(
                edge.enabled
                for edge in layout.walk_edges
                if frozenset((edge.start_node_id, edge.end_node_id)) in bridge_edges
            )


def test_shells_have_conservative_cross_block_separation() -> None:
    scenario = build_large_scenario(31)
    assert scenario.layout is not None
    shells = plan_shell_records(scenario)
    assert len(shells) == 144
    for record in shells:
        footprint = record.footprint
        assert footprint is not None
        assert record.collision is True
        assert footprint.actor_name == record.actor_name
        assert footprint.block_id == record.actor_name.removeprefix(
            "GEN_BP_DISTRICT_BUILDING_"
        ).rsplit("_", 1)[0]
        assert footprint.edge_index >= 0
        assert footprint.tangent_half_extent > 0.0
        assert footprint.normal_half_extent > 0.0
        block = scenario.layout.block_by_id(footprint.block_id)
        assert point_in_polygon(footprint.position, block.footprint)
        assert box_inside_block(block, footprint.bounds)
        edge = next(edge for edge in edge_frames(block) if edge.index == footprint.edge_index)
        expected_yaw = math.degrees(math.atan2(edge.outward[1], edge.outward[0]))
        assert math.isclose((footprint.yaw_deg - expected_yaw) % 360.0, 0.0, abs_tol=1e-6)

    # Collision uses the conservative axis-aligned footprint, not pivot-center
    # spacing.  Empty edge gaps may remain, but no two solid envelopes overlap.
    for index, first in enumerate(shells):
        first_box = first.footprint.bounds
        for second in shells[index + 1 :]:
            second_box = second.footprint.bounds
            separated = (
                first_box[2] <= second_box[0]
                or second_box[2] <= first_box[0]
                or first_box[3] <= second_box[1]
                or second_box[3] <= first_box[1]
            )
            assert separated


def test_shell_footprints_keep_graph_edges_buffered() -> None:
    scenario = build_large_scenario(31)
    assert scenario.layout is not None
    shells = plan_shell_records(scenario)
    routes = enabled_routes(scenario.layout)
    bridge_routes = enabled_bridge_routes(scenario.layout)
    for record in shells:
        footprint = record.footprint
        assert footprint is not None
        # Clearance is evaluated against the measured world-axis envelope and
        # actual route segments, not a pivot-centered circumscribed radius.
        assert box_clear_of_routes(footprint.bounds, routes, 400.0)
        assert box_clear_of_routes(footprint.bounds, bridge_routes, 500.0)


def test_route_segment_clearance_is_checked_between_nodes() -> None:
    _communicator, renderer = _spawn("medium")
    assert renderer.layout is not None
    routes = enabled_routes(renderer.layout)
    assert routes
    start, end = routes[0][0], routes[0][1]
    midpoint = ((start[0] + end[0]) / 2.0, (start[1] + end[1]) / 2.0)
    assert minimum_route_distance(midpoint, routes) < _ROUTE_CLEARANCE_CM


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
