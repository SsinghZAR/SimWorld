"""Focused offline tests for riverside_market_large_v1."""

from __future__ import annotations

from dataclasses import replace

from benchmark.venue_meetup.layout import WalkEdge
from benchmark.venue_meetup.template_validation import validate_layout
from benchmark.venue_meetup.templates.riverside_market import (
    EAST_X,
    MAP_TEMPLATE_ID,
    WEST_X,
    build_district_layout,
    build_fixed_scenario,
    layout_with_bridges_disabled,
)


def _bank_of(node_x: float) -> str:
    if node_x < 0.0:
        return "west"
    if node_x > 0.0:
        return "east"
    raise AssertionError(f"node sits on barrier centerline: x={node_x}")


def test_template_id_max_steps_and_agent_walk_nodes() -> None:
    scenario = build_fixed_scenario(seed=31)
    assert scenario.map_template_id == MAP_TEMPLATE_ID
    assert scenario.max_steps == 128
    assert scenario.layout is not None
    by_id = {agent.agent_id: agent for agent in scenario.agents}
    assert by_id["agent_0"].spawn_slot == "civic_plaza_spawn"
    assert by_id["agent_0"].walk_node_id == "spawn_civic_plaza"
    assert by_id["agent_0"].zone_id == "zone_west"
    assert by_id["agent_1"].spawn_slot == "transit_forecourt_spawn"
    assert by_id["agent_1"].walk_node_id == "spawn_transit_forecourt"
    assert by_id["agent_1"].zone_id == "zone_east"
    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        assert agent.walk_node_id != agent.spawn_slot
        node = scenario.layout.node_by_id(agent.walk_node_id)
        assert node.kind == "spawn"


def test_twelve_venues_six_blocks_and_district_span() -> None:
    scenario = build_fixed_scenario(seed=31)
    layout = scenario.layout
    assert layout is not None
    assert len(scenario.venues) == 12
    assert len({venue.venue_id for venue in scenario.venues}) == 12
    assert len({venue.position for venue in scenario.venues}) == 12
    assert len(layout.blocks) >= 6
    assert {block.block_id for block in layout.blocks} == {
        "block_nw_civic",
        "block_w_market",
        "block_sw_residential",
        "block_ne_transit",
        "block_e_waterfront",
        "block_se_hotel",
    }
    span_cm = EAST_X - WEST_X
    assert span_cm >= 70000.0
    west_venues = [venue for venue in scenario.venues if venue.zone_id == "zone_west"]
    east_venues = [venue for venue in scenario.venues if venue.zone_id == "zone_east"]
    assert len(west_venues) == 6
    assert len(east_venues) == 6
    assert 6 <= len(scenario.landmarks) <= 8


def test_venues_derive_pose_and_region_from_frontages() -> None:
    scenario = build_fixed_scenario(seed=31)
    layout = scenario.layout
    assert layout is not None
    venues_by_slot = {venue.slot_id: venue for venue in scenario.venues}
    assert len(layout.frontages) == 12
    for frontage in layout.frontages:
        assert frontage.venue_slot_id is not None
        venue = venues_by_slot[frontage.venue_slot_id]
        assert venue.position == frontage.position
        assert venue.yaw_deg == frontage.yaw_deg
        assert venue.region.center == frontage.meeting_region.center
        assert venue.region.radius == frontage.meeting_region.radius
        block = layout.block_by_id(frontage.block_id)
        assert frontage.frontage_id in block.frontage_ids


def test_validate_layout_paths_from_agent_walk_nodes_to_frontages() -> None:
    scenario = build_fixed_scenario(seed=31)
    layout = scenario.layout
    assert layout is not None
    assert layout == build_district_layout()

    walk_node_ids = [agent.walk_node_id for agent in scenario.agents]
    assert walk_node_ids == ["spawn_civic_plaza", "spawn_transit_forecourt"]
    assert not [node for node in layout.walk_nodes if node.kind == "frontage"]
    walk_node_id_set = {node.node_id for node in layout.walk_nodes}
    for frontage in layout.frontages:
        assert frontage.frontage_id not in walk_node_id_set
        assert frontage.approach_node_id is not None
        assert frontage.approach_node_id in walk_node_id_set

    approach_ids = [frontage.approach_node_id for frontage in layout.frontages]
    assert len(approach_ids) == 12
    required_paths = [
        (walk_node_id, approach_id)
        for walk_node_id in walk_node_ids
        for approach_id in approach_ids
        if walk_node_id is not None and approach_id is not None
    ]
    assert len(required_paths) == 24
    for start_id, end_id in required_paths:
        assert layout.is_reachable(start_id, end_id)
        assert layout.shortest_path(start_id, end_id) is not None
    validate_layout(layout, required_paths=required_paths)


def test_exactly_two_bridge_edges_and_no_other_barrier_crossing() -> None:
    layout = build_district_layout()
    bridge_edges = [edge for edge in layout.walk_edges if edge.route_kind == "bridge"]
    assert len(bridge_edges) == 2
    assert {edge.enabled for edge in bridge_edges} == {True}

    primary = next(
        edge
        for edge in bridge_edges
        if {edge.start_node_id, edge.end_node_id}
        == {"bridge_primary_west", "bridge_primary_east"}
    )
    secondary = next(
        edge
        for edge in bridge_edges
        if {edge.start_node_id, edge.end_node_id}
        == {"bridge_secondary_west", "bridge_secondary_east"}
    )
    assert primary.length_cm < secondary.length_cm

    nodes = {node.node_id: node for node in layout.walk_nodes}
    for edge in layout.walk_edges:
        start_bank = _bank_of(nodes[edge.start_node_id].position[0])
        end_bank = _bank_of(nodes[edge.end_node_id].position[0])
        if start_bank != end_bank:
            assert edge.route_kind == "bridge"
            assert edge in bridge_edges


def test_disabling_both_bridges_partitions_west_and_east() -> None:
    layout = build_district_layout()
    partitioned = layout_with_bridges_disabled(layout)
    assert all(not edge.enabled for edge in partitioned.walk_edges if edge.route_kind == "bridge")
    assert partitioned.is_reachable("spawn_civic_plaza", "swk_nw_cafe")
    assert partitioned.is_reachable("spawn_transit_forecourt", "swk_ne_shop")
    assert not partitioned.is_reachable("spawn_civic_plaza", "swk_ne_shop")
    assert not partitioned.is_reachable("spawn_transit_forecourt", "swk_nw_cafe")
    assert partitioned.shortest_path("spawn_civic_plaza", "swk_se_shop") is None

    # Manual disable via replace must match the helper.
    manual = replace(
        layout,
        walk_edges=tuple(
            WalkEdge(
                start_node_id=edge.start_node_id,
                end_node_id=edge.end_node_id,
                length_cm=edge.length_cm,
                enabled=False if edge.route_kind == "bridge" else edge.enabled,
                route_kind=edge.route_kind,
            )
            for edge in layout.walk_edges
        ),
    )
    assert manual.reachable_nodes("spawn_civic_plaza") == partitioned.reachable_nodes(
        "spawn_civic_plaza"
    )


def test_primary_bridge_is_shorter_cross_bank_route_than_secondary() -> None:
    layout = build_district_layout()
    # Cross-bank trip prefers the short primary bridge when both are open.
    path = layout.shortest_path("spawn_civic_plaza", "swk_ne_shop")
    assert path is not None
    assert "bridge_primary_west" in path or "bridge_primary_east" in path
    assert "bridge_secondary_west" not in path
    assert "bridge_secondary_east" not in path


def test_barrier_props_use_catalog_road_blocker() -> None:
    scenario = build_fixed_scenario(seed=31)
    barrier_props = [
        prop
        for venue in scenario.venues
        for prop in venue.props
        if prop.semantic == "canal/rail barrier"
    ]
    assert len(barrier_props) >= 6
    assert all(prop.asset_key == "RoadBlocker_C" for prop in barrier_props)
