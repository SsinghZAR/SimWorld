"""Scenario generation for Venue Meetup."""

from __future__ import annotations

import random
from dataclasses import dataclass, replace
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


@dataclass(frozen=True)
class HiddenProfileSpec:
    """Two-agent hidden-profile information design for a template size.

    Current limitation: ``num_agents`` must be 2. Venue counts of 4, 8, and 12
    are supported (``central_square_v0``, ``station_quarter_medium_v1``,
    ``riverside_market_large_v1``) as long as each agent zone has at least
    ``min_venues_per_zone`` venues.

    Role budget after zone assignment and optimum-side choice:

    - exactly one group-feasible optimum in the optimum agent's zone
    - exactly one optimum-zone decoy (shares the optimum agent's own hard need,
      lacks the partner need)
    - at least ``min_dependent_traps`` attractive traps in the dependent zone
      (partner need yes, optimum-agent need no)
    - every remaining venue is a non-feasible distractor
    """

    num_agents: int = 2
    min_venues_per_zone: int = 2
    min_dependent_traps: int = 2
    hard_keys: tuple[str, str] = HIDDEN_PROFILE_HARD_KEYS

    def validate_shape(self, *, num_agents: int, num_venues: int, zone_counts: dict[str, int]) -> None:
        """Reject unsupported agent counts or undersized zone partitions."""

        if num_agents != self.num_agents:
            raise ValueError(
                f"hidden_profile mode currently supports exactly {self.num_agents} agents, got {num_agents}"
            )
        if self.num_agents != 2:
            raise ValueError("HiddenProfileSpec currently hard-limits num_agents to 2")
        if len(zone_counts) != 2:
            raise ValueError(f"hidden_profile expects exactly two zones, got {zone_counts}")
        if any(count < self.min_venues_per_zone for count in zone_counts.values()):
            raise ValueError(
                f"hidden_profile requires at least {self.min_venues_per_zone} venues per zone, got {zone_counts}"
            )
        if num_venues < 2 * self.min_venues_per_zone:
            raise ValueError(
                f"hidden_profile needs at least {2 * self.min_venues_per_zone} venues, got {num_venues}"
            )
        # optimum + decoy in one zone; >= min traps in the other.
        if num_venues < 2 + self.min_dependent_traps:
            raise ValueError(
                "hidden_profile needs room for one optimum, one decoy, and "
                f"at least {self.min_dependent_traps} dependent-zone traps; got {num_venues} venues"
            )


def hidden_profile_spec_for(num_venues: int) -> HiddenProfileSpec:
    """Return the two-agent spec used for a template's venue count.

    The role construction is size-agnostic: larger templates simply receive more
    non-feasible distractors after the fixed optimum / decoy / trap budget.
    """

    del num_venues  # count is validated later against the resolved zone partition
    return HiddenProfileSpec()


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


def _nearest_spawn_zones(venues: list[Venue], agents: list[AgentSpec]) -> tuple[dict[str, str], dict[str, str]]:
    """Legacy central-square assignment: each venue joins the nearest agent's zone."""

    agent_zones = {agent.agent_id: f"zone_{agent.agent_id}" for agent in agents}
    venue_zones: dict[str, str] = {}
    for venue in venues:
        nearest = min(
            agents,
            key=lambda agent: (venue.position[0] - agent.position[0]) ** 2 + (venue.position[1] - agent.position[1]) ** 2,
        )
        venue_zones[venue.venue_id] = agent_zones[nearest.agent_id]
    return venue_zones, agent_zones


def _resolve_hidden_profile_zones(
    venues: list[Venue], agents: list[AgentSpec]
) -> tuple[dict[str, str], dict[str, str]]:
    """Resolve venue and agent zone ids for a hidden-profile overlay.

    Prefer each template's authored ``zone_id`` when every agent and every venue
    already has one (station / riverside). Otherwise use the legacy nearest-spawn
    fallback required by ``central_square_v0``.
    """

    agents_authored = all(agent.zone_id for agent in agents)
    venues_authored = all(venue.zone_id for venue in venues)
    if agents_authored and venues_authored:
        venue_zones = {venue.venue_id: str(venue.zone_id) for venue in venues}
        agent_zones = {agent.agent_id: str(agent.zone_id) for agent in agents}
        agent_zone_set = set(agent_zones.values())
        if len(agent_zone_set) != 2:
            raise ValueError(f"hidden_profile expects two distinct agent zones, got {agent_zones}")
        orphan = {zone for zone in venue_zones.values() if zone not in agent_zone_set}
        if orphan:
            raise ValueError(f"hidden_profile venues reference unknown zones {sorted(orphan)}")
        return venue_zones, agent_zones
    return _nearest_spawn_zones(venues, agents)


def _hidden_profile_properties(
    o_need: str,
    d_need: str,
    o_val: bool,
    d_val: bool,
    quiet: float,
    crowding: float,
    *,
    open: bool = True,
    reachable: bool = True,
) -> VenueProperties:
    """Build venue properties with the two discriminating needs set explicitly."""

    fields: dict[str, object] = dict(
        open=open,
        reachable=reachable,
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


def _distractor_properties(o_need: str, d_need: str, index: int, rng: random.Random) -> VenueProperties:
    """Non-group-feasible filler traits for venues beyond optimum/decoy/traps."""

    # Rotate through decoy classes so extra venues force inspection rather than
    # looking uniformly impossible. None may satisfy both discriminating needs
    # while remaining open and reachable.
    kind = index % 6
    if kind == 0:
        return _hidden_profile_properties(o_need, d_need, False, False, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.35))
    if kind == 1:
        return _hidden_profile_properties(o_need, d_need, False, False, rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.8))
    if kind == 2:
        return _hidden_profile_properties(o_need, d_need, True, False, rng.uniform(0.35, 0.85), rng.uniform(0.2, 0.7))
    if kind == 3:
        return _hidden_profile_properties(o_need, d_need, False, True, rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.8))
    if kind == 4:
        return _hidden_profile_properties(
            o_need,
            d_need,
            True,
            True,
            rng.uniform(0.7, 0.9),
            rng.uniform(0.1, 0.3),
            open=False,
        )
    return _hidden_profile_properties(o_need, d_need, True, False, rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.8))


def _retrait_entrances(venue: Venue, props: VenueProperties) -> list[Entrance]:
    """Re-derive entrance status from accessibility so visuals match hidden traits."""

    entrances = []
    for entrance in venue.entrances:
        if not props.open:
            status = "blocked"
        else:
            status = "accessible" if props.accessible else "stairs_only"
        entrances.append(Entrance(**{**entrance.__dict__, "status": status}))
    return entrances


def _build_trait_plan(
    *,
    o_zone_venues: list[Venue],
    d_zone_venues: list[Venue],
    o_need: str,
    d_need: str,
    spec: HiddenProfileSpec,
    rng: random.Random,
) -> dict[str, VenueProperties]:
    """Assign optimum / decoy / trap / distractor properties to every venue."""

    rng.shuffle(o_zone_venues)
    rng.shuffle(d_zone_venues)
    optimum_v, decoy_v = o_zone_venues[0], o_zone_venues[1]
    trap_venues = d_zone_venues[: spec.min_dependent_traps]
    distractor_venues = o_zone_venues[2:] + d_zone_venues[spec.min_dependent_traps :]

    trait_plan: dict[str, VenueProperties] = {
        optimum_v.venue_id: _hidden_profile_properties(
            o_need, d_need, True, True, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)
        ),
        decoy_v.venue_id: _hidden_profile_properties(
            o_need, d_need, True, False, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.3)
        ),
    }
    for trap_index, trap_v in enumerate(trap_venues):
        if trap_index == 0:
            trait_plan[trap_v.venue_id] = _hidden_profile_properties(
                o_need, d_need, False, True, rng.uniform(0.7, 0.9), rng.uniform(0.1, 0.35)
            )
        else:
            trait_plan[trap_v.venue_id] = _hidden_profile_properties(
                o_need, d_need, False, True, rng.uniform(0.2, 0.45), rng.uniform(0.55, 0.8)
            )
    for distractor_index, distractor_v in enumerate(distractor_venues):
        trait_plan[distractor_v.venue_id] = _distractor_properties(o_need, d_need, distractor_index, rng)
    return trait_plan


def _build_hidden_profile(scenario: Scenario, rng: random.Random, num_agents: int) -> Scenario:
    """Overlay a hidden-profile information structure on a template's geometry.

    Produces an instance that is provably NOT solvable by either agent alone:
    a unique group-feasible optimum sits in one agent's zone (so the partner
    depends on a report for it), the optimum-zone agent also has a decoy it cannot
    distinguish from the optimum using only its own need, and the partner's whole
    zone is infeasible for the group. See notes.md sections 3-5. The instance is
    checked against these invariants before it is returned.
    """

    spec = hidden_profile_spec_for(len(scenario.venues))
    # Fail first on the documented two-agent limitation before template shape checks.
    if num_agents != spec.num_agents:
        raise ValueError(
            f"hidden_profile mode currently supports exactly {spec.num_agents} agents, got {num_agents}"
        )

    venues = list(scenario.venues)
    agents = list(scenario.agents)[:num_agents]
    if len(agents) < num_agents:
        raise ValueError(f"template provides {len(scenario.agents)} agents but num_agents={num_agents}")

    venue_zones, agent_zones = _resolve_hidden_profile_zones(venues, agents)
    zone_counts: dict[str, int] = {}
    for zone in venue_zones.values():
        zone_counts[zone] = zone_counts.get(zone, 0) + 1

    spec.validate_shape(num_agents=num_agents, num_venues=len(venues), zone_counts=zone_counts)

    # Pick which agent's zone holds the optimum; the other agent is "dependent".
    optimum_agent = rng.choice(agents)
    dependent_agent = next(agent for agent in agents if agent.agent_id != optimum_agent.agent_id)
    # Assign the two hard needs (which agent needs which discriminating trait).
    needs = list(spec.hard_keys)
    rng.shuffle(needs)
    o_need, d_need = needs[0], needs[1]

    o_zone = agent_zones[optimum_agent.agent_id]
    d_zone = agent_zones[dependent_agent.agent_id]
    if o_zone == d_zone:
        raise ValueError("hidden_profile agents must occupy distinct zones")
    o_zone_venues = [v for v in venues if venue_zones[v.venue_id] == o_zone]
    d_zone_venues = [v for v in venues if venue_zones[v.venue_id] == d_zone]

    trait_plan = _build_trait_plan(
        o_zone_venues=o_zone_venues,
        d_zone_venues=d_zone_venues,
        o_need=o_need,
        d_need=d_need,
        spec=spec,
        rng=rng,
    )
    if set(trait_plan) != {venue.venue_id for venue in venues}:
        raise AssertionError("hidden_profile trait plan must cover every venue")

    rebuilt_venues = []
    for venue in venues:
        props = trait_plan[venue.venue_id]
        rebuilt_venues.append(
            replace(
                venue,
                properties=props,
                entrances=_retrait_entrances(venue, props),
                zone_id=venue_zones[venue.venue_id],
            )
        )

    rebuilt_agents = []
    for agent in agents:
        need = o_need if agent.agent_id == optimum_agent.agent_id else d_need
        # ``replace`` keeps authored walk_node_id / spawn geometry intact.
        rebuilt_agents.append(
            replace(
                agent,
                private_constraint=HIDDEN_PROFILE_CONSTRAINT_TEXT[need],
                private_requirement_keys=[need],
                zone_id=agent_zones[agent.agent_id],
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
    _assert_hidden_profile(built, optimum_agent.agent_id, dependent_agent.agent_id, o_need, d_need, spec)
    return built


def _assert_hidden_profile(
    scenario: Scenario,
    optimum_agent_id: str,
    dependent_agent_id: str,
    o_need: str,
    d_need: str,
    spec: HiddenProfileSpec,
) -> None:
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
    partner_req = Requirement(key=d_need, weight=1.0)
    o_zone_self_ok = [v for v in scenario.venues if v.zone_id == o_zone and satisfies(v, own_req, scenario)]
    if len(o_zone_self_ok) < 2:
        raise AssertionError("optimum-agent's zone needs a decoy that also satisfies its own need")
    decoys = [
        v
        for v in o_zone_self_ok
        if v.venue_id != optimum.venue_id and not satisfies(v, partner_req, scenario)
    ]
    if not decoys:
        raise AssertionError("optimum-zone decoy must share the own hard need but lack the partner need")
    # Attractive traps: dependent zone venues that look good under the dependent need alone.
    traps = [
        v
        for v in dep_zone_venues
        if satisfies(v, partner_req, scenario) and not satisfies(v, own_req, scenario)
    ]
    if len(traps) < spec.min_dependent_traps:
        raise AssertionError(
            f"dependent zone needs at least {spec.min_dependent_traps} attractive traps, found {len(traps)}"
        )
    # The decisive cross fact: optimum satisfies the partner's need, which the
    # optimum agent does not personally require (-> other-regarding sharing).
    if d_need == o_need or not satisfies(optimum, partner_req, scenario):
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
