"""Focused offline tests for station_quarter_medium_v1."""

from __future__ import annotations

from benchmark.venue_meetup.scenario import AgentSpec, scenario_from_dict
from benchmark.venue_meetup.template_validation import validate_layout
from benchmark.venue_meetup.templates.station_quarter import (
    MAP_TEMPLATE_ID,
    build_district_layout,
    build_fixed_scenario,
)


def test_agent_walk_node_ids_are_layout_spawn_nodes_not_spawn_slots() -> None:
    scenario = build_fixed_scenario(seed=21)
    assert scenario.map_template_id == MAP_TEMPLATE_ID
    assert scenario.layout is not None
    by_id = {agent.agent_id: agent for agent in scenario.agents}
    assert by_id["agent_0"].spawn_slot == "clock_tower_spawn"
    assert by_id["agent_0"].walk_node_id == "spawn_clock_tower"
    assert by_id["agent_1"].spawn_slot == "station_forecourt_spawn"
    assert by_id["agent_1"].walk_node_id == "spawn_station_forecourt"
    for agent in scenario.agents:
        assert agent.walk_node_id is not None
        assert agent.walk_node_id != agent.spawn_slot
        node = scenario.layout.node_by_id(agent.walk_node_id)
        assert node.kind == "spawn"


def test_agent_spec_parses_without_walk_node_id() -> None:
    """Legacy agent payloads omit walk_node_id and must still load."""

    agent = AgentSpec(
        agent_id="legacy",
        spawn_slot="clock_tower_spawn",
        position=(0.0, 0.0, 150.0),
        yaw_deg=0.0,
        private_constraint="legacy",
        private_requirement_keys=["accessible"],
    )
    assert agent.walk_node_id is None

    scenario = build_fixed_scenario(seed=21)
    payload = scenario.compact(include_hidden=True)
    for agent_payload in payload["agents"]:
        agent_payload.pop("walk_node_id", None)
    restored = scenario_from_dict(payload)
    assert all(agent.walk_node_id is None for agent in restored.agents)


def test_eight_distinct_venues_and_four_city_blocks() -> None:
    scenario = build_fixed_scenario(seed=21)
    assert scenario.layout is not None
    assert len(scenario.venues) == 8
    assert len({venue.venue_id for venue in scenario.venues}) == 8
    assert len({venue.position for venue in scenario.venues}) == 8
    assert len(scenario.layout.blocks) == 4
    assert {block.block_id for block in scenario.layout.blocks} == {
        "block_nw",
        "block_ne",
        "block_sw",
        "block_se",
    }


def test_region_centers_match_named_frontages() -> None:
    scenario = build_fixed_scenario(seed=21)
    layout = scenario.layout
    assert layout is not None
    venues_by_slot = {venue.slot_id: venue for venue in scenario.venues}
    assert len(layout.frontages) == 8
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
    scenario = build_fixed_scenario(seed=21)
    layout = scenario.layout
    assert layout is not None
    assert layout == build_district_layout()

    walk_node_ids = [agent.walk_node_id for agent in scenario.agents]
    assert walk_node_ids == ["spawn_clock_tower", "spawn_station_forecourt"]
    assert not [node for node in layout.walk_nodes if node.kind == "frontage"]
    walk_node_id_set = {node.node_id for node in layout.walk_nodes}
    for frontage in layout.frontages:
        assert frontage.frontage_id not in walk_node_id_set
        assert frontage.approach_node_id is not None
        assert frontage.approach_node_id in walk_node_id_set

    approach_ids = [frontage.approach_node_id for frontage in layout.frontages]
    assert len(approach_ids) == 8
    required_paths = [
        (walk_node_id, approach_id)
        for walk_node_id in walk_node_ids
        for approach_id in approach_ids
        if walk_node_id is not None and approach_id is not None
    ]
    assert len(required_paths) == 16
    for start_id, end_id in required_paths:
        assert layout.is_reachable(start_id, end_id)
        assert layout.shortest_path(start_id, end_id) is not None
    validate_layout(layout, required_paths=required_paths)
