"""Focused offline tests for layout-graph walk route planning."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.district_dressing import plan_shell_records
from benchmark.venue_meetup.district_geometry import DistrictShellFootprint
from benchmark.venue_meetup.layout import DistrictLayout, Frontage, MeetingRegion, WalkEdge, WalkNode
from benchmark.venue_meetup.navigation import (
    Obstacle,
    building_obstacles,
    plan_layout_route,
    plan_path,
    segment_clear,
    select_walk_planner,
)
from benchmark.venue_meetup.templates.central_square import build_fixed_scenario as build_central
from benchmark.venue_meetup.templates.riverside_market import (
    build_district_layout as build_riverside_layout,
)
from benchmark.venue_meetup.templates.riverside_market import (
    build_fixed_scenario as build_riverside,
)
from benchmark.venue_meetup.templates.riverside_market import layout_with_bridges_disabled
from benchmark.venue_meetup.templates.station_quarter import (
    build_district_layout as build_station_layout,
)
from benchmark.venue_meetup.templates.station_quarter import (
    build_fixed_scenario as build_station,
)
from tests.venue_meetup._district_geometry_oracle import (
    disc_chain,
    enabled_bridge_routes,
    enabled_routes,
)


def test_station_routes_are_graph_backed() -> None:
    scenario = build_station(seed=21)
    layout = scenario.layout
    assert layout is not None
    assert layout == build_station_layout()

    agent = next(agent for agent in scenario.agents if agent.agent_id == "agent_0")
    assert agent.walk_node_id == "spawn_clock_tower"
    assert select_walk_planner(layout=layout, walk_node_id=agent.walk_node_id) == "layout_graph"

    venue = next(venue for venue in scenario.venues if venue.slot_id == "ne_market_shop")
    route = plan_layout_route(layout, agent.walk_node_id, venue_slot_id=venue.slot_id)
    assert route is not None
    assert route.node_ids[0] == "spawn_clock_tower"
    frontage = next(frontage for frontage in layout.frontages if frontage.venue_slot_id == venue.slot_id)
    assert frontage.approach_node_id is not None
    assert route.node_ids[-1] == frontage.approach_node_id
    assert frontage.frontage_id not in route.node_ids
    assert layout.node_by_id(frontage.approach_node_id).position != frontage.meeting_region.center
    assert frontage.access_path and frontage.access_path[-1] == frontage.meeting_region.center
    assert all(edge.waypoints for edge in layout.walk_edges)
    assert route.access_path == frontage.access_path
    assert route.end_node_id == frontage.approach_node_id
    assert route.frontage_id == "front_ne_market_shop"
    assert route.graph_distance_cm > 0.0
    assert route.access_distance_cm > 0.0
    assert route.total_distance_cm == route.graph_distance_cm + route.access_distance_cm
    assert len(route.route_kinds) == len(route.node_ids) - 1
    assert route.used_bridge is False
    assert "bridge" not in route.route_kinds
    # Waypoints preserve each authored edge polyline in route order.
    nodes = {node.node_id: node for node in layout.walk_nodes}
    edges = {
        frozenset((edge.start_node_id, edge.end_node_id)): edge
        for edge in layout.walk_edges
    }
    expected_waypoints = []
    for left, right in zip(route.node_ids, route.node_ids[1:]):
        edge = edges[frozenset((left, right))]
        edge_waypoints = edge.waypoints if edge.start_node_id == left else tuple(reversed(edge.waypoints))
        expected_waypoints.extend((*edge_waypoints, nodes[right].position))
    assert route.waypoints == tuple(expected_waypoints)

    via_frontage = plan_layout_route(
        layout,
        agent.walk_node_id,
        frontage_id="front_ne_market_shop",
    )
    assert via_frontage == route


def test_riverside_cross_bank_uses_primary_bridge_only() -> None:
    scenario = build_riverside(seed=31)
    layout = scenario.layout
    assert layout is not None
    assert layout == build_riverside_layout()

    start = "spawn_civic_plaza"
    route = plan_layout_route(layout, start, venue_slot_id="ne_transit_shop")
    assert route is not None
    assert route.used_bridge is True
    assert "bridge" in route.route_kinds
    assert route.route_kinds.count("bridge") == 1
    assert "bridge_primary_west" in route.node_ids or "bridge_primary_east" in route.node_ids
    assert "bridge_secondary_west" not in route.node_ids
    assert "bridge_secondary_east" not in route.node_ids
    frontage = next(frontage for frontage in layout.frontages if frontage.venue_slot_id == "ne_transit_shop")
    assert frontage.approach_node_id is not None
    assert route.node_ids[-1] == frontage.approach_node_id
    assert frontage.frontage_id not in route.node_ids
    assert layout.node_by_id(frontage.approach_node_id).position != frontage.meeting_region.center
    assert frontage.access_path and frontage.access_path[-1] == frontage.meeting_region.center
    assert all(edge.waypoints for edge in layout.walk_edges)
    assert route.graph_distance_cm == layout.path_length_cm(start, frontage.approach_node_id)


def test_riverside_unreachable_when_both_bridges_disabled() -> None:
    layout = layout_with_bridges_disabled()
    assert select_walk_planner(layout=layout, walk_node_id="spawn_civic_plaza") == "layout_graph"
    route = plan_layout_route(layout, "spawn_civic_plaza", venue_slot_id="ne_transit_shop")
    assert route is None
    # Same-bank destinations remain reachable without bridges.
    same_bank = plan_layout_route(layout, "spawn_civic_plaza", venue_slot_id="nw_civic_cafe")
    assert same_bank is not None
    assert same_bank.used_bridge is False


def test_disabled_bridges_are_absent_from_district_geometry() -> None:
    layout = layout_with_bridges_disabled()
    bridge_edges = tuple(edge for edge in layout.walk_edges if edge.route_kind == "bridge")
    enabled_edges = tuple(edge for edge in layout.walk_edges if edge.enabled)

    assert bridge_edges and all(not edge.enabled for edge in bridge_edges)
    assert enabled_routes(layout) == tuple(layout.edge_polyline(edge) for edge in enabled_edges)
    assert enabled_bridge_routes(layout) == ()
    assert layout.is_reachable("spawn_civic_plaza", "swk_nw_cafe")
    assert not layout.is_reachable("spawn_civic_plaza", "swk_ne_shop")


def test_central_square_lacks_layout_and_falls_back() -> None:
    scenario = build_central(seed=7)
    assert scenario.layout is None
    assert all(agent.walk_node_id is None for agent in scenario.agents)
    assert select_walk_planner(layout=None, walk_node_id=None) == "obstacle_astar"
    assert select_walk_planner(layout=None, walk_node_id="spawn_clock_tower") == "obstacle_astar"
    empty = DistrictLayout(layout_id="empty")
    assert select_walk_planner(layout=empty, walk_node_id="spawn") == "obstacle_astar"


def _test_shell_footprint(
    half_extents: tuple[float, float],
    position: tuple[float, float],
) -> DistrictShellFootprint:
    return DistrictShellFootprint(
        actor_name="GEN_BP_DISTRICT_BUILDING_fixture_00",
        asset_key="BP_Building_05_C",
        block_id="fixture",
        edge_index=0,
        position=position,
        yaw_deg=0.0,
        scale=(1.0, 1.0, 1.0),
        half_extents=half_extents,
        tangent_half_extent=max(half_extents),
        normal_half_extent=min(half_extents),
    )


def _disc_obstacles(footprint: DistrictShellFootprint, clearance: float) -> tuple[Obstacle, ...]:
    return tuple(Obstacle(*values) for values in disc_chain(footprint.half_extents, footprint.position, clearance))


def _assert_disc_chains_match(actual, expected: tuple[Obstacle, ...]) -> None:
    assert len(actual) == len(expected)
    for observed, wanted in zip(actual, expected):
        assert observed.cx == pytest.approx(wanted.cx)
        assert observed.cy == pytest.approx(wanted.cy)
        assert observed.radius == pytest.approx(wanted.radius)


@pytest.mark.parametrize(
    ("half_extents", "position"),
    (
        ((120.0, 40.0), (10.0, -20.0)),
        ((40.0, 120.0), (-100.0, 70.0)),
    ),
)
def test_shell_obstacles_cover_horizontal_and_vertical_aabb_samples(
    half_extents: tuple[float, float],
    position: tuple[float, float],
) -> None:
    footprint = _test_shell_footprint(half_extents, position)
    caller_clearance = 20.0
    obstacles = _disc_obstacles(footprint, caller_clearance)
    assert obstacles

    hx, hy = half_extents
    cx, cy = position
    samples = []
    for fraction in (0.0, 0.125, 0.25, 0.375, 0.5, 0.625, 0.75, 0.875, 1.0):
        samples.extend(
            (
                (cx - hx + 2.0 * hx * fraction, cy - hy),
                (cx - hx + 2.0 * hx * fraction, cy + hy),
                (cx - hx, cy - hy + 2.0 * hy * fraction),
                (cx + hx, cy - hy + 2.0 * hy * fraction),
            )
        )
    assert all(any(obstacle.contains(sample) for obstacle in obstacles) for sample in samples)

    expected_radius = min(hx, hy) + caller_clearance
    assert all(obstacle.radius == pytest.approx(expected_radius) for obstacle in obstacles)
    repeated = _disc_obstacles(footprint, caller_clearance)
    assert obstacles == repeated
    centers = tuple((obstacle.cx, obstacle.cy) for obstacle in obstacles)
    if hx >= hy:
        assert all(y == pytest.approx(cy) for _, y in centers)
        assert tuple(x for x, _ in centers) == tuple(sorted(x for x, _ in centers))
        assert centers[0][0] == pytest.approx(cx - hx)
        assert centers[-1][0] == pytest.approx(cx + hx)
    else:
        assert all(x == pytest.approx(cx) for x, _ in centers)
        assert tuple(y for _, y in centers) == tuple(sorted(y for _, y in centers))
        assert centers[0][1] == pytest.approx(cy - hy)
        assert centers[-1][1] == pytest.approx(cy + hy)


def test_dense_shell_footprints_are_tiled_in_fallback_obstacles() -> None:
    caller_clearance = 500.0
    for scenario in (build_station(seed=21), build_riverside(seed=31)):
        shell_records = plan_shell_records(scenario)
        assert shell_records
        obstacles = building_obstacles(scenario, clearance=caller_clearance)
        expected_shells = 64 if len(scenario.venues) == 8 else 144
        prefix_count = len(scenario.venues) + len(scenario.landmarks)
        assert len(shell_records) == expected_shells
        shell_obstacles = obstacles[prefix_count:]
        expected_shell_obstacles = tuple(
            obstacle
            for record in shell_records
            for obstacle in _disc_obstacles(record.footprint, caller_clearance)
        )
        _assert_disc_chains_match(shell_obstacles, expected_shell_obstacles)
        assert len(shell_obstacles) >= len(shell_records)
        offset = 0
        for record in shell_records:
            footprint = record.footprint
            assert footprint is not None
            chain = _disc_obstacles(footprint, caller_clearance)
            assert chain
            _assert_disc_chains_match(shell_obstacles[offset : offset + len(chain)], chain)
            assert all(
                obstacle.radius == pytest.approx(min(footprint.half_extents) + caller_clearance)
                for obstacle in chain
            )
            offset += len(chain)
        assert offset == len(shell_obstacles)


@pytest.mark.parametrize(
    ("half_extents", "position"),
    (
        ((120.0, 40.0), (10.0, -20.0)),
        ((40.0, 120.0), (-100.0, 70.0)),
    ),
)
def test_shell_chain_preserves_parallel_clear_corridor_and_blocks_shell(
    half_extents: tuple[float, float],
    position: tuple[float, float],
) -> None:
    footprint = _test_shell_footprint(half_extents, position)
    caller_clearance = 20.0
    obstacles = list(_disc_obstacles(footprint, caller_clearance))
    hx, hy = half_extents
    cx, cy = position
    if hx >= hy:
        corridor_start = (cx - hx - 500.0, cy + hy + caller_clearance + 1.0)
        corridor_goal = (cx + hx + 500.0, cy + hy + caller_clearance + 1.0)
        through_start = (cx - hx - 500.0, cy)
        through_goal = (cx + hx + 500.0, cy)
    else:
        corridor_start = (cx + hx + caller_clearance + 1.0, cy - hy - 500.0)
        corridor_goal = (cx + hx + caller_clearance + 1.0, cy + hy + 500.0)
        through_start = (cx, cy - hy - 500.0)
        through_goal = (cx, cy + hy + 500.0)

    assert segment_clear(corridor_start, corridor_goal, obstacles)
    assert plan_path(corridor_start, corridor_goal, obstacles, cell=20.0) == [corridor_goal]
    assert not segment_clear(through_start, through_goal, obstacles)


def test_unknown_references_raise_clear_errors() -> None:
    layout = build_station_layout()
    with pytest.raises(ValueError, match="Unknown walk node_id"):
        plan_layout_route(layout, "missing_spawn", venue_slot_id="nw_market_cafe")
    with pytest.raises(ValueError, match="Unknown venue_slot_id"):
        plan_layout_route(layout, "spawn_clock_tower", venue_slot_id="no_such_slot")
    with pytest.raises(ValueError, match="Unknown frontage_id"):
        plan_layout_route(layout, "spawn_clock_tower", frontage_id="no_such_frontage")
    with pytest.raises(ValueError, match="exactly one of venue_slot_id or frontage_id"):
        plan_layout_route(layout, "spawn_clock_tower")
    with pytest.raises(ValueError, match="exactly one of venue_slot_id or frontage_id"):
        plan_layout_route(
            layout,
            "spawn_clock_tower",
            venue_slot_id="nw_market_cafe",
            frontage_id="front_nw_market_cafe",
        )


def test_frontage_without_walk_node_raises() -> None:
    layout = DistrictLayout(
        layout_id="orphan_frontage",
        frontages=(
            Frontage(
                frontage_id="orphan",
                block_id="b",
                position=(0.0, 0.0, 0.0),
                yaw_deg=0.0,
                entrance_point=(0.0, 0.0, 0.0),
                meeting_region=MeetingRegion(center=(0.0, 0.0), radius=100.0),
                venue_slot_id="orphan_slot",
            ),
        ),
        walk_nodes=(WalkNode("spawn", (0.0, 0.0), "spawn"),),
        walk_edges=(),
    )
    with pytest.raises(ValueError, match="has no approach_node_id and no matching walk node"):
        plan_layout_route(layout, "spawn", venue_slot_id="orphan_slot")


def test_unreachable_returns_none_without_raising() -> None:
    layout = DistrictLayout(
        layout_id="split",
        frontages=(
            Frontage(
                frontage_id="front_east",
                block_id="east",
                position=(10.0, 0.0, 0.0),
                yaw_deg=0.0,
                entrance_point=(10.0, 0.0, 0.0),
                meeting_region=MeetingRegion(center=(10.0, 0.0), radius=50.0),
                venue_slot_id="east_slot",
            ),
        ),
        walk_nodes=(
            WalkNode("spawn", (0.0, 0.0), "spawn"),
            WalkNode("front_east", (10.0, 0.0), "frontage"),
            WalkNode("island", (99.0, 99.0), "intersection"),
        ),
        walk_edges=(WalkEdge("island", "front_east", 1.0),),
    )
    assert plan_layout_route(layout, "spawn", venue_slot_id="east_slot") is None
