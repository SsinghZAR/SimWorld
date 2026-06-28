"""Scenario generation for Venue Meetup."""

from __future__ import annotations

import random
from dataclasses import replace
from typing import Callable

from benchmark.venue_meetup.scenario import AgentSpec, Entrance, Requirement, Scenario, Venue, VenueProperties
from benchmark.venue_meetup.scoring import satisfies, score_venue
from benchmark.venue_meetup.templates import TEMPLATE_BUILDERS

# Hidden-profile constants. The two discriminating *hard* needs; the optimum
# satisfies both, the decoy satisfies only the optimum-zone agent's need, and the
# traps satisfy only the dependent agent's need. See notes.md sections 3-5.
HIDDEN_PROFILE_HARD_KEYS: tuple[str, str] = ("accessible", "food_drink")
HIDDEN_PROFILE_CONSTRAINT_TEXT: dict[str, str] = {
    "accessible": "I need step-free access and cannot use stairs.",
    "food_drink": "I strongly prefer a place with food or drink available.",
}

PRIVATE_CONSTRAINTS: tuple[tuple[str, list[str]], ...] = (
    ("I need step-free access and cannot use stairs.", ["accessible"]),
    ("I strongly prefer food or drink and a quiet place.", ["food_drink", "quiet"]),
    ("I need shelter and want to avoid crowded places.", ["shelter", "uncrowded"]),
    ("I prefer staying near transit, but only if the venue is open.", ["near_transit", "open"]),
    ("I need enough capacity for the whole group.", ["capacity"]),
)

# Spawn slots clustered in the plaza centre (Unreal cm), each facing roughly
# toward the middle so agents can see the surrounding venues. z is human-ish
# eye height; movement is kinematic so agents keep this height.
SPAWN_RING: tuple[tuple[float, float, float, float], ...] = (
    (-600.0, -600.0, 150.0, 45.0),
    (600.0, 600.0, 150.0, -135.0),
    (-600.0, 600.0, 150.0, -45.0),
    (600.0, -600.0, 150.0, 135.0),
    (0.0, -850.0, 150.0, 90.0),
)


def _template_builder(template_id: str | None) -> Callable[[int], Scenario]:
    if template_id is None:
        return TEMPLATE_BUILDERS["central_square_v0"]
    if template_id not in TEMPLATE_BUILDERS:
        raise KeyError(f"Unknown template_id {template_id!r}; expected one of {sorted(TEMPLATE_BUILDERS)}")
    return TEMPLATE_BUILDERS[template_id]


def _randomize_properties(venues: list[Venue], rng: random.Random) -> list[Venue]:
    """Randomize hidden facts while preserving V0 generation constraints."""

    best_index = rng.randrange(len(venues))
    false_positive_index = (best_index + 1 + rng.randrange(len(venues) - 1)) % len(venues)
    closed_index = (best_index + 2 + rng.randrange(len(venues) - 1)) % len(venues)
    randomized: list[Venue] = []
    for index, venue in enumerate(venues):
        props = venue.properties
        if index == best_index:
            props = VenueProperties(
                open=True,
                reachable=True,
                capacity=max(4, props.capacity),
                accessible=True,
                shelter=True,
                food_drink=True,
                quiet_score=round(rng.uniform(0.68, 0.9), 2),
                crowding_score=round(rng.uniform(0.1, 0.35), 2),
                near_transit=props.near_transit,
            )
        elif index == false_positive_index:
            props = VenueProperties(
                open=True,
                reachable=True,
                capacity=max(2, props.capacity),
                accessible=False,
                shelter=props.shelter,
                food_drink=True,
                quiet_score=round(rng.uniform(0.35, 0.62), 2),
                crowding_score=round(rng.uniform(0.55, 0.85), 2),
                near_transit=True,
            )
        elif index == closed_index:
            props = replace(props, open=False, reachable=True, quiet_score=round(rng.uniform(0.45, 0.75), 2))
        else:
            props = VenueProperties(
                open=True,
                reachable=True,
                capacity=props.capacity,
                accessible=rng.random() > 0.35,
                shelter=props.shelter,
                food_drink=rng.random() > 0.35,
                quiet_score=round(rng.uniform(0.35, 0.85), 2),
                crowding_score=round(rng.uniform(0.15, 0.75), 2),
                near_transit=props.near_transit,
            )

        entrances = []
        for entrance in venue.entrances:
            status = entrance.status
            if index == closed_index:
                status = "blocked"
            elif not props.accessible:
                status = "stairs_only"
            elif props.open:
                status = "accessible"
            entrances.append(Entrance(**{**entrance.__dict__, "status": status}))
        randomized.append(replace(venue, properties=props, entrances=entrances))
    return randomized


def _agent_specs(num_agents: int, rng: random.Random) -> list[AgentSpec]:
    if num_agents < 2:
        raise ValueError("Venue Meetup requires at least two agents.")
    constraints = list(PRIVATE_CONSTRAINTS)
    rng.shuffle(constraints)
    agents: list[AgentSpec] = []
    for index in range(num_agents):
        x, y, z, yaw = SPAWN_RING[index % len(SPAWN_RING)]
        constraint, keys = constraints[index % len(constraints)]
        agents.append(
            AgentSpec(
                agent_id=f"agent_{index}",
                spawn_slot=f"spawn_{index}",
                position=(x, y, z),
                yaw_deg=yaw,
                private_constraint=constraint,
                private_requirement_keys=list(keys),
            )
        )
    return agents


def _hidden_profile_zones(venues: list[Venue], agents: list[AgentSpec]) -> dict[str, str]:
    """Assign each venue to the nearest agent spawn's zone (expects a 2-2 split)."""

    zone_of = {agent.agent_id: f"zone_{agent.agent_id}" for agent in agents}
    assignment: dict[str, str] = {}
    for venue in venues:
        nearest = min(
            agents,
            key=lambda agent: (venue.position[0] - agent.position[0]) ** 2 + (venue.position[1] - agent.position[1]) ** 2,
        )
        assignment[venue.venue_id] = zone_of[nearest.agent_id]
    counts: dict[str, int] = {}
    for zone in assignment.values():
        counts[zone] = counts.get(zone, 0) + 1
    if sorted(counts.values()) != [2, 2]:
        raise ValueError(f"hidden_profile expects a balanced 2-2 venue split by spawn proximity, got {counts}")
    return assignment


def _hidden_profile_properties(o_need: str, d_need: str, o_val: bool, d_val: bool, quiet: float, crowding: float) -> VenueProperties:
    """Build venue properties with the two discriminating needs set explicitly."""

    fields: dict[str, object] = dict(
        open=True,
        reachable=True,
        capacity=6,
        accessible=False,
        shelter=True,
        food_drink=False,
        quiet_score=round(quiet, 2),
        crowding_score=round(crowding, 2),
        near_transit=False,
    )
    fields[o_need] = o_val
    fields[d_need] = d_val
    return VenueProperties(**fields)  # type: ignore[arg-type]


def _retrait_entrances(venue: Venue, props: VenueProperties) -> list[Entrance]:
    """Re-derive entrance status from accessibility so visuals match hidden traits."""

    entrances = []
    for entrance in venue.entrances:
        status = "accessible" if props.accessible else "stairs_only"
        entrances.append(Entrance(**{**entrance.__dict__, "status": status}))
    return entrances


def _build_hidden_profile(scenario: Scenario, rng: random.Random, num_agents: int) -> Scenario:
    """Overlay a hidden-profile information structure on a template's geometry.

    Produces an instance that is provably NOT solvable by either agent alone:
    a unique group-feasible optimum sits in one agent's zone (so the partner
    depends on a report for it), the optimum-zone agent also has a decoy it cannot
    distinguish from the optimum using only its own need, and the partner's whole
    zone is infeasible for it. See notes.md sections 3-5. The instance is checked
    against these invariants before it is returned.
    """

    if num_agents != 2:
        raise ValueError("hidden_profile mode currently supports exactly 2 agents")
    venues = list(scenario.venues)
    if len(venues) != 4:
        raise ValueError("hidden_profile mode currently expects exactly 4 venues")
    agents = list(scenario.agents)[:2]

    zone_assignment = _hidden_profile_zones(venues, agents)
    zone_of_agent = {agent.agent_id: f"zone_{agent.agent_id}" for agent in agents}

    # Pick which agent's zone holds the optimum; the other agent is "dependent".
    optimum_agent = rng.choice(agents)
    dependent_agent = next(agent for agent in agents if agent.agent_id != optimum_agent.agent_id)
    # Assign the two hard needs (which agent needs which discriminating trait).
    needs = list(HIDDEN_PROFILE_HARD_KEYS)
    rng.shuffle(needs)
    o_need, d_need = needs[0], needs[1]

    o_zone = zone_of_agent[optimum_agent.agent_id]
    d_zone = zone_of_agent[dependent_agent.agent_id]
    o_zone_venues = [v for v in venues if zone_assignment[v.venue_id] == o_zone]
    d_zone_venues = [v for v in venues if zone_assignment[v.venue_id] == d_zone]
    rng.shuffle(o_zone_venues)
    rng.shuffle(d_zone_venues)
    optimum_v, decoy_v = o_zone_venues[0], o_zone_venues[1]
    trap_quiet_v, trap_noisy_v = d_zone_venues[0], d_zone_venues[1]

    # Trait assignment (o_need/d_need are accessible/food_drink booleans):
    #  - optimum:    O need Y, D need Y, quiet  -> only venue feasible for BOTH
    #  - decoy:      O need Y, D need N, quiet  -> ties optimum on O's own need
    #  - trap_quiet: O need N, D need Y, quiet  -> dependent agent's private best
    #  - trap_noisy: O need N, D need Y, noisy  -> worse trap
    trait_plan = {
        optimum_v.venue_id: _hidden_profile_properties(o_need, d_need, True, True, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)),
        decoy_v.venue_id: _hidden_profile_properties(o_need, d_need, True, False, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)),
        trap_quiet_v.venue_id: _hidden_profile_properties(o_need, d_need, False, True, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.35)),
        trap_noisy_v.venue_id: _hidden_profile_properties(o_need, d_need, False, True, rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.8)),
    }
    rebuilt_venues = []
    for venue in venues:
        props = trait_plan[venue.venue_id]
        rebuilt_venues.append(
            replace(venue, properties=props, entrances=_retrait_entrances(venue, props), zone_id=zone_assignment[venue.venue_id])
        )

    rebuilt_agents = []
    for agent in agents:
        need = o_need if agent.agent_id == optimum_agent.agent_id else d_need
        rebuilt_agents.append(
            replace(
                agent,
                private_constraint=HIDDEN_PROFILE_CONSTRAINT_TEXT[need],
                private_requirement_keys=[need],
                zone_id=zone_of_agent[agent.agent_id],
            )
        )

    requirements = [
        Requirement(key="open", weight=2.0, hard=True, description="Venue must be open."),
        Requirement(key="reachable", weight=2.0, hard=True, description="Venue must be physically reachable."),
        Requirement(key="accessible", weight=2.0, hard=True, description="Step-free access is required by a visitor."),
        Requirement(key="food_drink", weight=2.0, hard=True, description="Food or drink is required by a visitor."),
        Requirement(key="quiet", weight=1.0, description="Quiet venues are preferred (soft tie-breaker)."),
    ]
    soft_weights = {"quiet_threshold": 0.65, "crowding_threshold": 0.5}

    built = replace(
        scenario,
        scenario_id=f"{scenario.map_template_id}_hp_seed_{scenario.seed}_n{num_agents}",
        venues=rebuilt_venues,
        agents=rebuilt_agents,
        requirements=requirements,
        soft_weights=soft_weights,
    )
    _assert_hidden_profile(built, optimum_agent.agent_id, dependent_agent.agent_id, o_need, d_need)
    return built


def _assert_hidden_profile(scenario: Scenario, optimum_agent_id: str, dependent_agent_id: str, o_need: str, d_need: str) -> None:
    """Fail loudly if a generated instance is not a true hidden profile."""

    feasible = [v for v in scenario.venues if not score_venue(v, scenario).hard_failures]
    if len(feasible) != 1:
        raise AssertionError(f"hidden_profile must have exactly one group-feasible optimum, found {len(feasible)}")
    optimum = feasible[0]
    o_zone = next(a.zone_id for a in scenario.agents if a.agent_id == optimum_agent_id)
    d_zone = next(a.zone_id for a in scenario.agents if a.agent_id == dependent_agent_id)
    if optimum.zone_id != o_zone:
        raise AssertionError("optimum must sit in the optimum-agent's zone")
    # Dependent agent's whole zone is infeasible -> it must rely on the partner.
    dep_zone_venues = [v for v in scenario.venues if v.zone_id == d_zone]
    if any(not score_venue(v, scenario).hard_failures for v in dep_zone_venues):
        raise AssertionError("dependent agent's zone must contain no group-feasible venue")
    # Optimum agent cannot distinguish optimum from a decoy using only its own need.
    own_req = Requirement(key=o_need, weight=1.0)
    o_zone_self_ok = [v for v in scenario.venues if v.zone_id == o_zone and satisfies(v, own_req, scenario)]
    if len(o_zone_self_ok) < 2:
        raise AssertionError("optimum-agent's zone needs a decoy that also satisfies its own need")
    # The decisive cross fact: optimum satisfies the partner's need, which the
    # optimum agent does not personally require (-> other-regarding sharing).
    if d_need == o_need or not satisfies(optimum, Requirement(key=d_need, weight=1.0), scenario):
        raise AssertionError("optimum must satisfy the partner's distinct need (other-regarding fact)")


def generate_scenario(
    *,
    seed: int,
    template_id: str | None = None,
    num_agents: int = 2,
    randomize: bool = True,
    hidden_profile: bool = False,
) -> Scenario:
    """Generate a deterministic venue-meetup scenario.

    When ``hidden_profile`` is set, the template geometry is reused but venue
    traits, agent constraints, and partition zones are overlaid to form a
    hidden-profile information structure (see notes.md). This takes precedence
    over ``randomize``.
    """

    rng = random.Random(seed)
    builder = _template_builder(template_id)
    scenario = builder(seed)
    if hidden_profile:
        built = _build_hidden_profile(scenario, rng, num_agents)
        return replace(built, seed=seed)
    venues = list(scenario.venues)
    if randomize:
        # Randomized variants (eval matrix): shuffle hidden facts and synthesize
        # plaza-ring spawns/constraints for diversity across seeds.
        venues = _randomize_properties(venues, rng)
        agents = _agent_specs(num_agents, rng)
    else:
        # Fixed scenario (smoke/eval-by-template): honor the template's authored
        # spawns and constraints exactly, padding from the ring only if the
        # template defines fewer agents than requested.
        agents = list(scenario.agents)[:num_agents]
        if len(agents) < num_agents:
            agents = agents + _agent_specs(num_agents, rng)[len(agents):]
    requirements = list(scenario.requirements)
    if not any(requirement.key == "uncrowded" for requirement in requirements):
        requirements.append(Requirement(key="uncrowded", weight=0.75, description="Low crowding is preferred."))
    if not any(requirement.key == "capacity" for requirement in requirements):
        requirements.append(Requirement(key="capacity", weight=0.75, description="Venue should fit the full group."))
    return replace(
        scenario,
        scenario_id=f"{scenario.map_template_id}_seed_{seed}_n{num_agents}",
        seed=seed,
        venues=venues,
        agents=agents,
        requirements=requirements,
    )


def evaluation_matrix(templates: list[str] | None = None, seeds: range | list[int] = range(10), agent_counts: tuple[int, ...] = (2, 3)) -> list[Scenario]:
    """Return the small-evaluation scenario matrix."""

    template_ids = templates or list(TEMPLATE_BUILDERS.keys())
    scenarios: list[Scenario] = []
    for template_id in template_ids:
        for seed in seeds:
            for num_agents in agent_counts:
                scenarios.append(generate_scenario(seed=int(seed), template_id=template_id, num_agents=num_agents, randomize=True))
    return scenarios
