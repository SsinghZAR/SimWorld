"""End-to-end scheduler/evidence/communication tests with a deterministic engine."""

from dataclasses import replace

import pytest

from benchmark.venue_meetup._core.action_space import VenueAgentTurn
from benchmark.venue_meetup.generator import generate_scenario
from benchmark.venue_meetup.scoring import score_venue
from benchmark.venue_meetup.targeted_scripted import TargetedScriptedPolicy
from benchmark.venue_meetup.timed_navigation import (
    TimedRoute,
    advance_route,
    finish_route,
)
from benchmark.venue_meetup.timing import TimingConfig
from benchmark.venue_meetup.varied_profiles import varied_profile
from tests.venue_meetup._targeted_adapter import OfflineTargetedEnv


def make_env(**kwargs):
    built = varied_profile(generate_scenario(seed=7, template_id="rosebank_grid_5x5_v0", hidden_profile=True))
    return OfflineTargetedEnv(built, **kwargs)


def test_no_early_evidence_or_intermediate_rewards():
    env = make_env()
    agent, partner = env.agent_ids
    venue = next(v for v in env.scenario.venues if v.zone_id == env.scenario.agents[0].zone_id)
    env.positions[agent] = venue.region.center
    action = VenueAgentTurn(choice=3, target_interactable_id=f"{venue.venue_id}__access")
    observations, rewards, done, info = env.step({agent: action, partner: VenueAgentTurn(choice=4, message="Hello")})
    assert rewards == {} and "scores" not in info and not done
    assert observations[agent]["known_venue_evidence"] == {}
    assert observations[agent]["group_chat"][0]["content"] == "Hello"
    assert env.ready_agent_ids == (partner,)
    assert observations[partner]["known_venue_evidence"] == {}
    observations, rewards, done, info = env.step({partner: VenueAgentTurn()})
    assert set(env.revealed_facts[agent][venue.venue_id]) == {"accessible", "reachable"}
    assert observations[partner]["known_venue_evidence"] == {}
    assert "own_activity" in observations[partner] and "activities_internal" not in observations[partner]
    assert observations[agent]["inspection_history"][0]["observed_tick"] == 2


def test_deadline_does_not_finish_inspection_and_does_not_expose_score():
    env = make_env(timing=TimingConfig(starts_at="17:59"))
    agent, partner = env.agent_ids
    venue = next(v for v in env.scenario.venues if v.zone_id == env.scenario.agents[0].zone_id)
    env.positions[agent] = venue.region.center
    env.step({agent: VenueAgentTurn(choice=3, target_interactable_id=f"{venue.venue_id}__meeting_area"),
              partner: VenueAgentTurn()})
    observations, rewards, done, info = env.step({partner: VenueAgentTurn()})
    assert done and not info["success"] and "scores" in info and rewards == {}
    assert observations[agent]["known_venue_evidence"] == {}
    assert all("scores" not in obs and "reward" not in obs for obs in observations.values())
    with pytest.raises(RuntimeError):
        env.step({})


def test_wrong_meetup_is_terminal_but_not_successful():
    env = make_env()
    wrong = next(v for v in env.scenario.venues if score_venue(v, env.scenario).hard_failures)
    env.positions = {agent: wrong.region.center for agent in env.agent_ids}
    _, _, done, info = env.step({agent: VenueAgentTurn() for agent in env.agent_ids})
    assert done and not info["success"] and info["scores"]["episode_score"] < 1


def test_completion_exactly_at_deadline_and_safety_cap_independence():
    env = make_env(timing=TimingConfig(starts_at="17:59"))
    optimum = next(v for v in env.scenario.venues if not score_venue(v, env.scenario).hard_failures)
    env.positions = {agent: optimum.region.center for agent in env.agent_ids}
    env.scenario = replace(env.scenario, max_steps=1)
    # A two-tick action cannot end the episode at the first tick even though both are present.
    agent, partner = env.agent_ids
    owner = next(a.agent_id for a in env.scenario.agents if a.zone_id == optimum.zone_id)
    other = partner if owner == agent else agent
    _, _, done, _ = env.step({owner: VenueAgentTurn(choice=3, target_interactable_id=f"{optimum.venue_id}__access"),
                             other: VenueAgentTurn()})
    assert not done
    _, _, done, info = env.step({other: VenueAgentTurn()})
    assert done and info["success"] and info["scores"]["episode_score"] == 1
    assert info["closing_clock"]["current_time"] == "18:00:00"


def test_information_constrained_scripted_episode():
    env = make_env()
    policy = TargetedScriptedPolicy()
    observations = env.reset()
    for _ in range(env.timing.max_ticks):
        available = {agent: obs for agent, obs in observations.items() if agent in env.ready_agent_ids}
        turns, _ = policy.act_all(available)
        observations, rewards, done, info = env.step(turns)
        if done:
            break
    assert info["success"], (env.step_index, env.bus.snapshot(), env.revealed_evidence)
    assert info["scores"]["episode_score"] == 1


def test_large_safety_cap_does_not_construct_a_legacy_clock():
    built = make_env().scenario
    env = OfflineTargetedEnv(replace(built, max_steps=10000))
    assert env.timing.snapshot(0)["current_time"] == "17:30:00"
    assert env.timing.max_ticks == 60


def test_invalid_venue_and_whole_venue_inspection_cost_time_without_facts():
    env = make_env()
    first, second = env.agent_ids
    _, _, done, info = env.step({first: VenueAgentTurn(choice=5, target_venue_id="missing"),
                                second: VenueAgentTurn(choice=3, target_venue_id="missing")})
    assert not done and env.step_index == 1
    assert info["actions"][first]["result"] == "NAVIGATE_FAILED"
    assert info["actions"][second]["result"] == "INSPECT_FAILED"
    assert not any(env.revealed_facts.values())


def test_waypoint_obstruction_after_region_entry_is_still_arrival():
    env = make_env()
    agent = env.agent_ids[0]
    venue = env.scenario.venues[0]
    route = TimedRoute(venue.venue_id, [[]], 1000, None, blocked=True)
    env.positions[agent] = venue.region.center
    result = finish_route(env, agent, route)
    assert result["arrived"] and result["result"] == "NAVIGATE_OK"
    env.positions[agent] = (1000000, 1000000)
    result = finish_route(env, agent, route)
    assert not result["arrived"] and result["result"] == "NAVIGATE_BLOCKED"


def test_long_busy_action_preserves_received_messages():
    env = make_env()
    receiver, sender = env.agent_ids
    env.scheduler.start(receiver, VenueAgentTurn(), 15)
    for index in range(15):
        observations, _, done, _ = env.step({sender: VenueAgentTurn(choice=4, message=f"Message {index}")})
        assert not done
    assert receiver in env.ready_agent_ids
    assert len(observations[receiver]["group_chat"]) == 15
    assert observations[receiver]["group_chat"][0]["content"] == "Message 0"


def test_early_region_arrival_keeps_advertised_travel_time():
    env = make_env()
    agent = env.agent_ids[0]
    venue = env.scenario.venues[0]
    route = TimedRoute(venue.venue_id, [[venue.region.center], [venue.region.center]], 4100, None)
    def blocked_at_destination(aid, point, *, last_venue):
        env.positions[aid] = venue.region.center
        return 3900, False
    env._walk_segment = blocked_at_destination
    assert not advance_route(env, agent, route, 0)
    assert route.arrived_early and not route.blocked
    assert len(route.chunks) == 2
    assert not advance_route(env, agent, route, 1)
    assert route.moved_cm == 3900
