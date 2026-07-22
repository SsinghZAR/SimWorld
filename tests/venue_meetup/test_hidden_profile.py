"""Offline hidden-profile generator invariants (no UE / network)."""

from __future__ import annotations

import pytest

from benchmark.venue_meetup.generator import (
    HiddenProfileSpec,
    generate_scenario,
    hidden_profile_spec_for,
)
from benchmark.venue_meetup.scoring import satisfies, score_venue
from benchmark.venue_meetup.scenario import Requirement
from benchmark.venue_meetup.templates import TEMPLATE_BUILDERS

HP_TEMPLATES: tuple[tuple[str, int, int], ...] = (
    ("central_square_v0", 4, 32),
    ("station_quarter_medium_v1", 8, 64),
    ("riverside_market_large_v1", 12, 128),
)
SEEDS = range(20)


def _generate(template_id: str, seed: int):
    return generate_scenario(
        seed=seed,
        template_id=template_id,
        num_agents=2,
        hidden_profile=True,
    )


@pytest.mark.parametrize(("template_id", "venue_count", "max_steps"), HP_TEMPLATES)
@pytest.mark.parametrize("seed", SEEDS)
def test_hidden_profile_invariants_across_templates(
    template_id: str, venue_count: int, max_steps: int, seed: int
) -> None:
    scenario = _generate(template_id, seed)
    spec = hidden_profile_spec_for(venue_count)

    assert scenario.map_template_id == template_id
    assert scenario.max_steps == max_steps
    assert len(scenario.venues) == venue_count
    assert len(scenario.agents) == 2
    assert scenario.scenario_id == f"{template_id}_hp_seed_{seed}_n2"

    # One group-feasible venue only.
    feasible = [v for v in scenario.venues if not score_venue(v, scenario).hard_failures]
    assert len(feasible) == 1
    optimum = feasible[0]

    agent_zones = {agent.agent_id: agent.zone_id for agent in scenario.agents}
    assert None not in agent_zones.values()
    assert len(set(agent_zones.values())) == 2
    assert all(venue.zone_id is not None for venue in scenario.venues)

    zone_counts: dict[str, int] = {}
    for venue in scenario.venues:
        zone_counts[venue.zone_id] = zone_counts.get(venue.zone_id, 0) + 1
    assert len(zone_counts) == 2
    assert all(count >= spec.min_venues_per_zone for count in zone_counts.values())
    assert set(zone_counts) == set(agent_zones.values())

    o_zone = optimum.zone_id
    d_zone = next(zone for zone in zone_counts if zone != o_zone)
    optimum_agent = next(agent for agent in scenario.agents if agent.zone_id == o_zone)
    dependent_agent = next(agent for agent in scenario.agents if agent.zone_id == d_zone)
    assert len(optimum_agent.private_requirement_keys) == 1
    assert len(dependent_agent.private_requirement_keys) == 1
    o_need = optimum_agent.private_requirement_keys[0]
    d_need = dependent_agent.private_requirement_keys[0]
    assert o_need != d_need
    assert {o_need, d_need} == set(spec.hard_keys)

    own_req = Requirement(key=o_need, weight=1.0)
    partner_req = Requirement(key=d_need, weight=1.0)

    # Partner dependence: dependent zone has no group-feasible venue.
    dep_zone_venues = [v for v in scenario.venues if v.zone_id == d_zone]
    assert all(score_venue(v, scenario).hard_failures for v in dep_zone_venues)

    # Decoy invariant: optimum zone has own-need match that lacks partner need.
    o_zone_self_ok = [v for v in scenario.venues if v.zone_id == o_zone and satisfies(v, own_req, scenario)]
    assert len(o_zone_self_ok) >= 2
    decoys = [
        v
        for v in o_zone_self_ok
        if v.venue_id != optimum.venue_id and not satisfies(v, partner_req, scenario)
    ]
    assert decoys
    assert satisfies(optimum, partner_req, scenario)

    traps = [
        v
        for v in dep_zone_venues
        if satisfies(v, partner_req, scenario) and not satisfies(v, own_req, scenario)
    ]
    assert len(traps) >= spec.min_dependent_traps

    # Every additional venue beyond optimum/decoy/min traps is non-feasible.
    assert all(score_venue(v, scenario).hard_failures for v in scenario.venues if v.venue_id != optimum.venue_id)


def test_num_agents_not_two_is_explicit_limitation() -> None:
    with pytest.raises(ValueError, match="exactly 2 agents"):
        generate_scenario(seed=0, template_id="central_square_v0", num_agents=3, hidden_profile=True)


def test_hidden_profile_spec_documents_two_agent_limit() -> None:
    spec = HiddenProfileSpec()
    assert spec.num_agents == 2
    assert spec.min_venues_per_zone == 2
    assert spec.min_dependent_traps == 2
    with pytest.raises(ValueError, match="exactly 2 agents"):
        spec.validate_shape(num_agents=3, num_venues=8, zone_counts={"a": 4, "b": 4})


@pytest.mark.parametrize("seed", SEEDS)
def test_central_legacy_nearest_spawn_zones(seed: int) -> None:
    """central_square_v0 has no authored zones; keep nearest-spawn zone_* ids."""

    base = TEMPLATE_BUILDERS["central_square_v0"](seed)
    assert all(agent.zone_id is None for agent in base.agents)
    assert all(venue.zone_id is None for venue in base.venues)

    scenario = _generate("central_square_v0", seed)
    assert {agent.zone_id for agent in scenario.agents} == {"zone_agent_0", "zone_agent_1"}
    assert {venue.zone_id for venue in scenario.venues} == {"zone_agent_0", "zone_agent_1"}
    counts = {}
    for venue in scenario.venues:
        counts[venue.zone_id] = counts.get(venue.zone_id, 0) + 1
    assert counts == {"zone_agent_0": 2, "zone_agent_1": 2}


@pytest.mark.parametrize(
    ("template_id", "expected_zones"),
    [
        ("station_quarter_medium_v1", {"zone_west", "zone_east"}),
        ("riverside_market_large_v1", {"zone_west", "zone_east"}),
    ],
)
@pytest.mark.parametrize("seed", (0, 7, 19))
def test_authored_zone_ids_preferred(template_id: str, expected_zones: set[str], seed: int) -> None:
    base = TEMPLATE_BUILDERS[template_id](seed)
    scenario = _generate(template_id, seed)
    assert {agent.zone_id for agent in scenario.agents} == expected_zones
    assert {venue.zone_id for venue in scenario.venues} == expected_zones
    for agent in scenario.agents:
        base_agent = next(a for a in base.agents if a.agent_id == agent.agent_id)
        assert agent.zone_id == base_agent.zone_id
        assert agent.walk_node_id == base_agent.walk_node_id
        assert agent.walk_node_id is not None
    for venue in scenario.venues:
        base_venue = next(v for v in base.venues if v.venue_id == venue.venue_id)
        assert venue.zone_id == base_venue.zone_id


@pytest.mark.parametrize("seed", SEEDS)
def test_central_agents_have_no_walk_node_id(seed: int) -> None:
    scenario = _generate("central_square_v0", seed)
    assert all(agent.walk_node_id is None for agent in scenario.agents)
