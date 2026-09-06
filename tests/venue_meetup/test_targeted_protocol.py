"""Contracts for targeted observations, varied needs and asynchronous tick costs."""

from dataclasses import asdict
from types import SimpleNamespace

import numpy as np
import pytest

from benchmark.venue_meetup._core.action_space import VenueAgentTurn, sanitize_turn
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.interaction_runtime import InteractionRuntime
from benchmark.venue_meetup.interactions import INTERACTION_KINDS, interaction_points
from benchmark.venue_meetup.prompt import build_targeted_system_prompt
from benchmark.venue_meetup.scoring import venue_decision_facts
from benchmark.venue_meetup.timed_navigation import plan_timed_route, route_chunks
from benchmark.venue_meetup.timing import ActionScheduler, TimingConfig
from benchmark.venue_meetup.varied_profiles import (
    NEED_TEXT,
    validate_varied_profile,
    varied_profile,
)


def scenario(seed=7, size=5):
    return varied_profile(generate_scenario(seed=seed, template_id=f"rosebank_grid_{size}x{size}_v0", hidden_profile=True))


def test_varied_profiles_across_scales_and_seeds():
    seen, counts = set(), set()
    for size in (3, 5, 7):
        for seed in range(24):
            built = scenario(seed, size)
            validate_varied_profile(built)
            for agent in built.agents:
                seen.update(agent.private_requirement_keys)
                counts.add(len(agent.private_requirement_keys))
            assert not (set(built.agents[0].private_requirement_keys) & set(built.agents[1].private_requirement_keys))
    assert seen == set(NEED_TEXT)
    assert counts == {1, 2}
    assert scenario().compact() == scenario().compact()


def test_clock_independent_busy_actions_and_deadline():
    timing = TimingConfig(starts_at="17:58")
    scheduler = ActionScheduler(["a", "b"], timing)
    scheduler.start("a", VenueAgentTurn(choice=5), 4)
    scheduler.start("b", VenueAgentTurn(choice=3), 1)
    assert set(scheduler.advance()) == {"b"}
    assert scheduler.ready == ("b",)
    assert scheduler.activity("a")["ticks_remaining"] == 3
    with pytest.raises(ValueError):
        scheduler.start("a", VenueAgentTurn(), 1)
    scheduler.start("b", VenueAgentTurn(choice=3), 2)
    assert scheduler.advance() == {}
    assert set(scheduler.advance()) == {"b"}
    scheduler.start("b", VenueAgentTurn(choice=4), 2)
    assert set(scheduler.advance()) == {"a"}
    assert "b" in scheduler.pending  # no early result for unfinished message
    assert scheduler.ready == ()
    assert timing.snapshot(scheduler.tick)["current_time"] == "18:00:00"
    with pytest.raises(RuntimeError):
        scheduler.advance()


@pytest.mark.parametrize("kwargs", [{"tick_seconds": 0}, {"travel_metres_per_tick": 0},
                                   {"travel_metres_per_tick": float("nan")}, {"starts_at": "18:00"}])
def test_invalid_timing(kwargs):
    with pytest.raises(ValueError):
        TimingConfig(**kwargs)


def test_navigation_chunks_keep_corners_and_cost_distance():
    chunks = route_chunks((0, 0), [(100, 0), (100, 100)], 75)
    assert chunks == [[(75, 0)], [(100, 0), (100, 50)], [(100, 100)]]
    assert TimingConfig().travel_ticks(16000) == 4
    assert TimingConfig().travel_ticks(16001) == 5


def test_layout_routes_from_spawns_and_between_venues():
    built = scenario()
    agent = built.agents[0]
    for origin in (None, built.venues[0]):
        node = agent.walk_node_id if origin is None else next(
            frontage.approach_node_id for frontage in built.layout.frontages if frontage.venue_slot_id == origin.slot_id)
        env = SimpleNamespace(scenario=built, timing=TimingConfig(),
                              _agent_walk_nodes={agent.agent_id: node},
                              get_agent_state=lambda aid: SimpleNamespace(actor_name=aid),
                              _actor_xy=lambda actor: agent.position[:2] if origin is None else origin.region.center,
                              _meeting_target=lambda aid, venue: venue.region.center,
                              _frontage_node_id_for_venue=lambda venue: None)
        for venue in built.venues:
            route = plan_timed_route(env, agent.agent_id, venue)
            assert len(route.chunks) == env.timing.travel_ticks(route.distance_cm)


def test_interaction_ids_and_masks_do_not_encode_feasibility():
    points = interaction_points(scenario())
    assert len(points) == 32
    colors = [np.array(point.mask_color) for point in points.values()]
    assert min(np.max(np.abs(a - b)) for i, a in enumerate(colors) for b in colors[i + 1:]) > 16
    alternate = interaction_points(scenario(8))
    assert [point.public() for point in points.values()] == [point.public() for point in alternate.values()]
    action = VenueAgentTurn.from_json({"choice": 3, "target_interactable_id": next(iter(points))})
    assert sanitize_turn(action).target_interactable_id == action.target_interactable_id


def interaction_fixture():
    built = scenario()
    agent = built.agents[0].agent_id
    venue = next(venue for venue in built.venues if venue.zone_id == built.agents[0].zone_id)
    mask = np.zeros((100, 200, 3), dtype=np.uint8)
    env = SimpleNamespace(scenario=built, agent_ids=built.agent_ids(), timing=TimingConfig(), step_index=3,
                          revealed_facts={}, revealed_evidence={}, inspected_venues=set(),
                          get_agent_state=lambda aid: SimpleNamespace(actor_name=aid),
                          _actor_xy=lambda actor: venue.region.center,
                          _can_inspect_zone=lambda aid, v: v.zone_id == built.agents[0].zone_id,
                          _capture_frames=lambda mode: {agent: mask},
                          _venue_facts=lambda v: venue_decision_facts(v, built.soft_weights))
    return InteractionRuntime(env), agent, venue, mask


def test_inspections_merge_only_selected_facts_and_fail_without_visibility():
    runtime, agent, venue, mask = interaction_fixture()
    for index, kind in enumerate(INTERACTION_KINDS):
        point = runtime.points[f"{venue.venue_id}__{kind.key}"]
        action = VenueAgentTurn(choice=3, target_interactable_id=point.interaction_id)
        failed = runtime.inspect(agent, action)
        assert failed["result"] == "INSPECT_FAILED" and "facts" not in failed
        mask[:] = 0
        mask[:10, :10] = point.mask_color[::-1]
        result = runtime.inspect(agent, action)
        assert result["result"] == "INSPECT_OK"
        assert set(result["facts"]) == set(kind.traits)
        assert set(runtime.env.revealed_facts[agent][venue.venue_id]) == {
            trait for previous in INTERACTION_KINDS[:index + 1] for trait in previous.traits}
        repeated = runtime.inspect(agent, action)
        assert repeated["repeat_internal"] and repeated["evidence"] == result["evidence"]
        mask[:] = 0
    assert runtime.env.revealed_facts.get("agent_1", {}) == {} if agent != "agent_1" else True
    assert runtime.inspect(agent, VenueAgentTurn(choice=3, target_venue_id=venue.venue_id))["result"] == "INSPECT_FAILED"


def test_prompt_configuration_and_no_runtime_rewards():
    prompt = build_targeted_system_prompt(asdict(TimingConfig()))
    assert '"tick_seconds": 30' in prompt
    assert "512 characters" in prompt
    assert "no intermediate rewards" in prompt
    assert "only when" in prompt and "exactly at closing" in prompt
    assert "Cooperative strategy addendum" not in prompt
    assert '"meeting_area": 3' in prompt
